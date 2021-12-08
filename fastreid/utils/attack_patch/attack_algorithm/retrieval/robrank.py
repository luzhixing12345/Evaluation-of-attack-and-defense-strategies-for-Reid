
import torch
import torch.nn.functional as F
import numpy as np
import functools
import os
from fastreid.utils.reid_patch import get_test_set, save_image,change_preprocess_image
import statistics
from scipy import stats
from fastreid.engine import DefaultTrainer
from fastreid.modeling.heads.build import build_feature_heads
from fastreid.utils.checkpoint import Checkpointer
device='cuda'
_LEGAL_ATTAKS_ = ('ES', 'QA', 'CA', 'SPQA', 'GTM', 'GTT', 'TMA', 'LTM')

def ROBRANK(cfg,query_data_loader,gallery_data_loader):
    '''
    l means the length of query set, we need to split the test set as query set and gallery set
    '''
    model = DefaultTrainer.build_model_main(cfg)#this model was used for later evaluations
    model.preprocess_image = change_preprocess_image(cfg) 
    model.heads = build_feature_heads(cfg)
    model.to(device)
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    rob = robrank(cfg,query_data_loader,gallery_data_loader,model)
    rob.run()

class robrank():
    def __init__(self,cfg,query_data_loader,gallery_data_loader,model) -> None:
        self.model=model
        self.query_data_loader=query_data_loader
        self.gallery_data_loader=gallery_data_loader
        self.attack_method = cfg.MODEL.ATTACKMETHOD
        self.pm=None
        if self.attack_method[-1]=='-' or self.attack_method[-1]=='+':
            self.pm=self.attack_method[-1]
            self.attack_method=self.attack_method[:-1]
            
        self.metric='N'
        self.kw={}
        self.kw['device'] = device
        self.kw['attack_type'] = self.attack_method
        self.kw['W']=1
        self.kw['M']=1
        self.kw['pm']=self.pm
        self.kw['metric']='N'
    
    def run(self):
        '''
        Note, all images must lie in [0,1]^D
        '''
        # prepare torche current batch of data
        gallery = self.recompute_valvecs()
        #gallery[0] -> features
        #gallery[1] -> targets
        # XXX: this is tricky, but we need it.

        # initialize attacker
        advrank = AdvRank(self.model,**self.kw)

        iterator = None

        if self.attack_method=='QA':
            iterator = self.query_data_loader 
            pos = 'adv_query'

            for id,data in enumerate(iterator):
                images = (data['images']/255.0).to(device)
                labels = data['targets'].to(device)
                path = data['img_paths']
                adv_data = advrank(images, labels, gallery)
                save_image(adv_data,path,pos)

        elif self.attack_method=='CA':
            iterator = self.gallery_data_loader
            pos = 'adv_bounding_box_test'

            for id,data in enumerate(iterator):
                images = (data['images']/255.0).to(device)
                labels = data['targets'].to(device)
                path = data['img_paths']
                advrank(images, labels, gallery)
    
    def recompute_valvecs(self):
        '''
        Compute embedding vectors for the whole validation dataset,
        in order to do image retrieval and evaluate the Recall@K scores,
        etc.
        '''
        with torch.no_grad():
            valvecs, vallabs = [], []
            for _, data in enumerate(self.gallery_data_loader):
                images = (data['images']/255.0).to(device)
                labels = data['targets'].to(device)
                features = self.model(images)
                features = F.normalize(features, p=2, dim=-1)
                valvecs.append(features.detach())
                vallabs.append(labels.detach())
            valvecs, vallabs = torch.cat(valvecs), torch.cat(vallabs)
        # XXX: in DDP mode the size of valvecs is 10000, looks correct
        # if str(self._distrib_type) in ('DistributedType.DDP', 'DistributedType.DDP2'):
        #    torch.distributed.barrier()
        #    sizes_slice = [torch.tensor(0).to(self.device)
        #                   for _ in range(torch.distributed.get_world_size())]
        #    size_slice = torch.tensor(len(valvecs)).to(self.device)
        #    torch.distributed.all_gather(sizes_slice, size_slice)
        #    print(sizes_slice)
        #    print(f'[torch.distributed.get_rank()]Shape:',
        #          valvecs.shape, vallabs.shape)
        # model._valvecs = valvecs
        # model._vallabs = vallabs
        #print('valvecs.shape = ',valvecs.shape)
        return (valvecs, vallabs)


class AdvRank:
    def __init__(self, model: torch.nn.Module, *,
                 eps: float = 4. / 255., alpha: float = 1. / 255., pgditer: int = 24,
                 attack_type: str = None,
                 M: int = None, W: int = None, pm: str = None,
                 verbose: bool = False, device: str = 'cuda',
                 metric: str = None):

        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.pgditer = pgditer
        self.attack_type = attack_type
        self.M = M
        self.W = W
        self.pm = pm
        self.verbose = verbose
        self.device = device
        self.metric = metric
        self.XI = 1.
    def __call__(self, images: torch.Tensor, labels: torch.Tensor,
               candi: tuple) -> tuple:
        '''
        Note, all images must lie in [0,1]^D
        '''
        # prepare the current batch of data
        assert(isinstance(images, torch.Tensor))
        images = images.detach()
        images_orig = images.clone()
        images = images.requires_grad_()
        labels = labels.to(self.device).view(-1)
        attack_type = self.attack_type
        

        with torch.no_grad():
            output, dist = self.outputdist(images, labels, candi)
            #print('dist.shape = ',dist.shape)
            #dist -> length of gallery set ->dist[i]: distance between the pic from query to one of gallery pics
            #print('candi[0].shape = ',candi[0].shape)
            #print('candi[1].shape = ',candi[1].shape)

            # dist.shape =  torch.Size([128, 17661])
            # candi[0].shape =  torch.Size([17661, 2048])
            # candi[1].shape =  torch.Size([17661])

            #candi[0] -> features ->candi[0][i]:batch_size x featrues_shape(2048 default)
            #candi[1] -> targets  ->candi[1][i]:batch_size


            #output_orig, dist_orig, summary_orig = self.eval_advrank(
            #    images, labels, candi, resample=True)
        self.qcsel = QCSelector(f'{self.attack_type}{self.pm}', self.M, self.W)(dist, candi)
        
        for iteration in range(self.pgditer):
            # >> prepare optimizer for SGD
            optim = torch.optim.SGD(self.model.parameters(), lr=1.)
            optimx = torch.optim.SGD([images], lr=1.)
            optim.zero_grad()
            optimx.zero_grad()
            output = self.forwardmetric(images)

            # calculate differentiable loss
            if (attack_type == 'ES'):
                if iteration == 0:
                    # avoid zero gradient
                    loss, _ = AdvRankLoss('ES', self.metric)(
                        output, output_orig + 1e-7)
                else:
                    loss, _ = AdvRankLoss('ES', self.metric)(
                        output, output_orig)

            elif (attack_type == 'FOA') and (self.M == 2):
                # >> reverse the inequalities (ordinary: d1 < d2, adversary: d1 > d2)
                embpairs, _ = self.qcsel
                loss, _ = AdvRankLoss('FOA2', self.metric)(
                    output, embpairs[:, 1, :], embpairs[:, 0, :])

            elif (attack_type == 'SPFOA') and (self.M == 2):
                embpairs, _, embgts, _ = self.qcsel
                loss, _ = AdvRankLoss('FOA2', self.metric)(
                    output, embpairs[:, 1, :], embpairs[:, 0, :])
                loss_sp, _ = AdvRankLoss(f'QA{self.pm}', self.metric)(
                    output, embgts, candi[0])
                loss = loss + self. XI * loss_sp

            elif (attack_type == 'FOA') and (self.M > 2):
                # >> enforce the random inequality set (di < dj for all i,j where i<j)
                embpairs, _ = self.qcsel
                loss, _ = AdvRankLoss('FOAX', self.metric)(output, embpairs)

            elif (attack_type == 'SPFOA') and (self.M > 2):
                embpairs, _, embgts, _ = self.qcsel
                loss, _ = AdvRankLoss('FOAX', self.metric)(output, embpairs)
                loss_sp, _ = AdvRankLoss(f'QA{self.pm}', self.metric)(
                    output, embgts, candi[0])
                self.update_xi(loss_sp)
                loss = loss + self.XI * loss_sp
            elif (attack_type == 'CA'):
                embpairs, _ = self.qcsel
                loss, _ = AdvRankLoss(f'CA{self.pm}', self.metric)(
                        output, embpairs, candi[0])
            elif (attack_type == 'QA'):
                embpairs, _ = self.qcsel
                # == enforce the target set of inequalities, while preserving the semantic

                loss, _ = AdvRankLoss('QA', self.metric)(
                        output, embpairs, candi[0], pm=self.pm)
            elif (attack_type == 'SPQA'):
                embpairs, _, embgts, _ = self.qcsel

                loss_qa, _ = AdvRankLoss('QA', self.metric)(
                    output, embpairs, candi[0], pm=self.pm)
                loss_sp, _ = AdvRankLoss('QA', self.metric)(
                    output, embgts, candi[0], pm='+')
                self.update_xi(loss_sp)
                loss = loss_qa + self.XI * loss_sp
            elif (attack_type == 'GTM'):
                ((emm, _), (emu, _), (ems, _)) = self.qcsel
                loss = AdvRankLoss('GTM', self.metric)(
                    output, emm, emu, ems, candi[0])
                # Note: greedy qc selection / resample harms performance
                # with torch.no_grad():
                #    if self.metric in ('C',):
                #        dist = 1 - output @ candi[0].t()
                #    elif self.metric in ('E', 'N'):
                #        dist = torch.cdist(output, candi[0])
                # self.qcsel = QCSelector('GTM', None, None)(dist, candi,
                #        self.dist_orig)
            elif (attack_type == 'GTT'):
                ((emm, idm), (emu, idum), (ems, _)) = self.qcsel
                loss = AdvRankLoss('GTT', self.metric)(
                    output, emm, emu, ems, candi[0])
            elif attack_type == 'TMA':
                (embrand, _) = self.qcsel
                loss = AdvRankLoss('TMA', self.metric)(output, embrand)
            elif attack_type == 'LTM':
                mask_same = (candi[1].view(1, -1) == labels.view(-1, 1))
                mask_same.scatter(1, self.loc_self.view(-1, 1), False)
                mask_diff = (candi[1].view(1, -1) != labels.view(-1, 1))
                if self.metric in ('E', 'N'):
                    dist = torch.cdist(output, candi[0])
                elif self.metric == 'C':
                    dist = 1 - output @ candi[0].t()
                maxdan = torch.stack([dist[i, mask_diff[i]].max()
                                   for i in range(dist.size(0))])
                mindap = torch.stack([dist[i, mask_same[i]].min()
                                   for i in range(dist.size(0))])
                loss = (maxdan - mindap).relu().sum()
            else:
                raise Exception("Unknown attack")
            loss.backward()
            
            # grad = images.grad
            # images = images + self.alpha *torch.sign(images.grad)
            # images = images.clone().detach()
            # images.requires_grad = True
            # images.grad = grad
            images.grad.data.copy_(self.alpha * torch.sign(images.grad))
 
            optimx.step()
            # L_infty constraint
            #images = torch.min(images, images_orig + self.eps)
            # L_infty constraint
            #images = torch.max(images, images_orig - self.eps)
            images = torch.clamp(images, min=0., max=1.)
            images = images.clone().detach()
            images.requires_grad = True
        
        print("change = ",images-images_orig)
        optim.zero_grad()
        optimx.zero_grad()
        return images
    def forwardmetric(self, images: torch.Tensor) -> torch.Tensor:
        '''
        metric-aware forwarding
        '''
        output = self.model(images)
        if self.metric in ('C', 'N'):
            return F.normalize(output)
        elif self.metric in ('E', ):
            return output

    def RankLossCandidateAttack(self, cs: torch.Tensor, Qs: torch.Tensor, Xs: torch.Tensor, *, pm: str):
        '''
        Computes the loss function for pure candidate attack

        Arguments:
            cs: size(batch, embedding_dim), embeddings of candidates.
            Qs: size(batch, W, embedding_dim), embedding of selected queries.
            Xs: size(testsize, embedding_dim), embedding of test set.
            pm: either '+' or '-'
        '''
        assert(cs.shape[1] == Qs.shape[2] == Xs.shape[1])
        NIter, W, D, NX = cs.shape[0], Qs.shape[1], Qs.shape[2], Xs.shape[0]
        losses, ranks = [], []
        for i in range(NIter):
            # == compute pairwise distance
            c = cs[i].view(1, D)  # [1, output_1]
            Q = Qs[i, :, :].view(W, D)  # [W, output_1]
            if self.metric == 'C':
                A = 1 - torch.mm(c, Q.t()).expand(NX, W)  # [candi_0, W]
                B = 1 - torch.mm(Xs, Q.t())  # [candi_0, W]
            elif self.metric in ('E', 'N'):
                A = (Q - c).norm(2, dim=1).expand(NX, W)  # [candi_0, W]
                B = torch.cdist(Xs, Q, p=2.0)
                # B2 = (Xs.view(NX, 1, D).expand(NX, W, D) -
                #     Q.view(1, W, D).expand(NX, W, D)).norm(2, dim=2)  # [candi_0, W]
                #assert((B-B2).abs().norm() < 1e-4)
            # == loss function
            if '+' == pm:
                loss = (A - B).clamp(min=0.).mean()
            elif '-' == pm:
                loss = (-A + B).clamp(min=0.).mean()
            losses.append(loss)
            # == compute the rank. Note, the > sign is correct
            rank = ((A > B).float().mean() * NX).item()
            ranks.append(rank)
        loss = torch.stack(losses).mean()
        rank = statistics.mean(ranks)
        return (loss, rank)

    
    def outputdist(self, images: torch.Tensor, labels: torch.Tensor,
                   candi: tuple) -> tuple:
        '''
        calculate output, and dist w.r.t. candidates.

        Note, this function does not differentiate. It's used for evaluation
        '''
    
        with torch.no_grad():
            if self.metric == 'C':
                output = self.forwardmetric(images)
                # [num_output_num, num_candidate]
                dist = 1 - output @ candi[0].t()
            elif self.metric in ('E', 'N'):
                output = self.forwardmetric(images)
                dist = torch.cdist(output, candi[0])
                # the memory requirement is insane
                # should use more efficient method for Euclidean distance
                # dist2 = []
                # for i in range(output.shape[0]):
                #    xq = output[i].view(1, -1)
                #    xqd = (candi[0] - xq).norm(2, dim=1).squeeze()
                #    dist2.append(xqd)
                # dist2 = torch.stack(dist2)  # [num_output_num, num_candidate]
                #assert((dist2 - dist).norm() < 1e-3)
            else:
                raise ValueError(self.metric)
            output_detach = output.clone().detach()
            dist_detach = dist.clone().detach()
        return (output_detach, dist_detach)

    def eval_advrank(self, images, labels, candi, *, resample=True) -> dict:
        '''
        evaluate original images or adversarial images for ranking
        `resample` is used for retaining selection for multiple times of evals.

        side effect:
            it sets self.qcsel when resample is toggled
        '''
        # evaluate original output and dist
        output, dist = self.outputdist(images, labels, candi)
        '''
        dist:  该img和所有其他图片的features计算距离
        output：该img的features
        '''
        attack_type = self.attack_type
        M, W = self.M, self.W

        # [[[ dispatch: qcselection and original evaluation ]]]
        # -> dispatch: ES
        if (attack_type == 'ES'):
            # select queries and candidates for ES
            if resample:
                self.qcsel = QCSelector('ES', M, W, False)(dist, candi)
                self.output_orig = output.clone().detach()
            output_orig = self.output_orig
            # evaluate the attack
            allLab = candi[1].cpu().numpy().squeeze()
            localLab = labels.cpu().numpy().squeeze()
            r_1, r_10, r_100 = [], [], []
            if resample:
                for i in range(dist.shape[0]):
                    agsort = dist[i].cpu().numpy().argsort()[1:]
                    rank = np.where(allLab[agsort] == localLab[i])[0].min()
                    r_1.append(rank == 0)
                    r_10.append(rank < 10)
                    r_100.append(rank < 100)
            else:
                # We are now evaluating adversarial examples
                # hence masking the query itself in this way
                for i in range(dist.shape[0]):
                    if self.metric == 'C':
                        loc = 1 - candi[0] @ output_orig[i].view(-1, 1)
                        loc = loc.flatten().argmin().cpu().numpy()
                    else:
                        loc = (candi[0] - output_orig[i]).norm(2, dim=1)
                        loc = loc.flatten().argmin().cpu().numpy()
                    dist[i][loc] = 1e38  # according to float32 range.
                    agsort = dist[i].cpu().numpy().argsort()[0:]
                    rank = np.where(allLab[agsort] == localLab[i])[0].min()
                    r_1.append(rank == 0)
                    r_10.append(rank < 10)
                    r_100.append(rank < 100)
            r_1, r_10, r_100 = 100 * \
                np.mean(r_1), 100 * np.mean(r_10), 100 * np.mean(r_100)
            loss, _ = AdvRankLoss('ES', self.metric)(output, output_orig)
            # summary
            summary_orig = {'loss': loss.item(), 'r@1': r_1,
                            'r@10': r_10, 'r@100': r_100}

        # -> dispatch: LTM
        elif attack_type == 'LTM':
            if resample:
                self.output_orig = output.clone().detach()
                self.loc_self = dist.argmin(dim=-1).view(-1)
            allLab = candi[1].cpu().numpy().squeeze()
            localLab = labels.cpu().numpy().squeeze()
            r_1 = []
            for i in range(dist.size(0)):
                dist[i][self.loc_self[i]] = 1e38
                argsort = dist[i].cpu().numpy().argsort()[0:]
                rank = np.where(allLab[argsort] == localLab[i])[0].min()
                r_1.append(rank == 0)
            r_1 = np.mean(r_1)
            # summary
            summary_orig = {'r@1': r_1}

        # -> dispatch: TMA
        elif attack_type == 'TMA':
            if resample:
                self.output_orig = output.clone().detach()
                self.qcsel = QCSelector('TMA', None, None)(dist, candi)
            (embrand, _) = self.qcsel
            cossim = F.cosine_similarity(output, embrand).mean().item()
            # summary
            summary_orig = {'Cosine-SIM': cossim}

        # -> dispatch: GTM
        elif (attack_type == 'GTM'):
            if resample:
                self.output_orig = output.clone().detach()
                self.dist_orig = dist.clone().detach()
                self.loc_self = self.dist_orig.argmin(dim=-1).view(-1)
                self.qcsel = QCSelector('GTM', None, None)(dist, candi,
                                                           self.loc_self)
            output_orig = self.output_orig
            # evaluate the attack
            allLab = candi[1].cpu().numpy().squeeze()
            localLab = labels.cpu().numpy().squeeze()
            r_1 = []
            # the process is similar to that for ES attack
            # except that we only evaluate recall at 1 (r_1)
            for i in range(dist.shape[0]):
                dist[i][self.loc_self[i]] = 1e38
                argsort = dist[i].cpu().numpy().argsort()[0:]
                rank = np.where(allLab[argsort] == localLab[i])[0].min()
                r_1.append(rank == 0)
            r_1 = np.mean(r_1)
            # summary
            summary_orig = {'r@1': r_1}

        # -> dispatch: GTT
        elif (attack_type == 'GTT'):
            if resample:
                self.output_orig = output.clone().detach()
                self.dist_orig = dist.clone().detach()
                self.loc_self = self.dist_orig.argmin(dim=-1).view(-1)
                self.qcsel = QCSelector('GTT', None, None)(
                    dist, candi, self.loc_self)
            dist[range(len(self.loc_self)), self.loc_self] = 1e38
            ((_, idm), (_, _), (_, _)) = self.qcsel
            re1 = (dist.argmin(dim=-1).view(-1) == idm).float().mean().item()
            dk = {}
            for k in (4,):
                topk = dist.topk(k, dim=-1, largest=False)[1]
                seq = [topk[:, j].view(-1) == idm for j in range(k)]
                idrecall = functools.reduce(torch.logical_or, seq)
                dk[f'retain@{k}'] = idrecall.float().mean().item()
            # summary
            summary_orig = {'ID-Retain@1': re1, **dk}

        # -> dispatch: FOA M=2
        elif (attack_type == 'FOA') and (M == 2):
            # select quries and candidates for FOA(M=2)
            if resample:
                self.qcsel = QCSelector('FOA', M, W)(dist, candi)
            embpairs, msample = self.qcsel
            # >> compute the (ordinary) loss on selected targets
            loss, acc = AdvRankLoss('FOA2', self.metric)(
                output, embpairs[:, 1, :], embpairs[:, 0, :])
            # summary
            summary_orig = {'loss': loss.item(), 'FOA2:Accuracy': acc}

        # -> dispatch: SPFOA M=2
        elif (attack_type == 'SPFOA') and (M == 2):
            if resample:
                self.qcsel = QCSelector('FOA', M, W, True)(dist, candi)
            embpairs, msample, embgts, mgtruth = self.qcsel
            loss, acc = AdvRankLoss('FOA2', self.metric)(
                output, embpairs[:, 1, :], embpairs[:, 0, :])
            loss_sp, rank_gt = AdvRankLoss(
                'QA+', self.metric)(output, embgts, candi[0], dist=dist, cidx=mgtruth)
            self.update_xi(loss_sp)
            loss = loss + self.XI * loss_sp
            # summary
            summary_orig = {'loss': loss.item(), 'loss_sp': loss_sp.item(),
                            'FOA2:Accuracy': acc, 'GT.mR': rank_gt / candi[0].size(0)}

        # -> dispatch: FOA M>2
        elif (attack_type == 'FOA') and (M > 2):
            if resample:
                self.qcsel = QCSelector('FOA', M, W)(dist, candi)
            embpairs, msample = self.qcsel
            loss, tau = AdvRankLoss('FOAX', self.metric)(output, embpairs)
            summary_orig = {'loss': loss.item(), 'FOA:tau': tau}

        # -> dispatch: SPFOA M>2
        elif (attack_type == 'SPFOA') and (M > 2):
            if resample:
                self.qcsel = QCSelector('FOA', M, W, True)(dist, candi)
            embpairs, msample, embgts, mgtruth = self.qcsel
            loss, tau = AdvRankLoss('FOAX', self.metric)(output, embpairs)
            loss_sp, rank_sp = AdvRankLoss(
                'QA+', self.metric)(output, embgts, candi[0], dist=dist, cidx=mgtruth)
            loss = loss + self.XI * loss_sp
            summary_orig = {'loss': loss.item(), 'loss_sp': loss_sp.item(),
                            'FOA:tau': tau, 'GT.mR': rank_sp / candi[0].size(0)}
        # -> dispatch: CA
        elif (attack_type == 'CA'):
            if resample:
                self.qcsel = QCSelector(f'CA{self.pm}', M, W)(dist, candi)
            embpairs, msamples = self.qcsel
            
            #return:
            #embpairs: msample 对应的features
            #msample : n x W(1)  n个随机的坐标
            
            loss, rank = AdvRankLoss(f'CA{self.pm}', self.metric)(
                output, embpairs, candi[0])
            mrank = rank / candi[0].shape[0]
            summary_orig = {'loss': loss.item(), f'CA{self.pm}:prank': mrank}
            print(summary_orig)

        # -> dispatch: QA
        elif (attack_type == 'QA'):
            if resample:
                self.qcsel = QCSelector(f'QA{self.pm}', M, W)(dist, candi)
            embpairs, msample = self.qcsel
            loss, rank_qa = AdvRankLoss(f'QA{self.pm}', self.metric)(
                output, embpairs, candi[0], dist=dist, cidx=msample)
            mrank = rank_qa / candi[0].shape[0]  # percentile ranking
            summary_orig = {'loss': loss.item(), f'QA{self.pm}:prank': mrank}

        # -> dispatch: SPQA
        elif (attack_type == 'SPQA'):
            if resample:
                self.qcsel = QCSelector(
                    f'QA{self.pm}', M, W, True)(dist, candi)
            embpairs, msample, embgts, mgtruth = self.qcsel
            loss_qa, rank_qa = AdvRankLoss(f'QA{self.pm}', self.metric)(
                output, embpairs, candi[0], dist=dist, cidx=msample)
            loss_sp, rank_sp = AdvRankLoss(
                'QA+', self.metric)(output, embgts, candi[0], dist=dist, cidx=mgtruth)
            self.update_xi(loss_sp)
            loss = loss_qa + self.XI * loss_sp
            mrank = rank_qa / candi[0].shape[0]
            mrankgt = rank_sp / candi[0].shape[0]
            summary_orig = {'loss': loss.item(), f'SPQA{self.pm}:prank': mrank,
                            f'SPQA{self.pm}:GTprank': mrankgt}

        # -> dispatch: N/A
        else:
            raise Exception("Unknown attack")
        # note: QCSelector results are stored in self.qcsel
        return output, dist, summary_orig
    def embShift(self, images: torch.Tensor, orig: torch.Tensor = None) -> torch.Tensor:
        '''
        barely performs the ES attack without any evaluation.
        used for adv training. [2002.11293]

        Returns the adversarial example.
        '''
        assert(isinstance(images, torch.Tensor))
        images = images.clone().detach().to(self.device)
        images_orig = images.clone().detach()
        images.requires_grad = True

        # evaluate original samples, and set self.qcsel
        with torch.no_grad():
            output_orig = orig if orig is not None else self.forwardmetric(
                images)
        # -> start PGD optimization
        self.model.eval()
        for iteration in range(self.pgditer):
            # >> prepare optimizer for SGD
            optim = torch.optim.SGD(self.model.parameters(), lr=0.)
            optimx = torch.optim.SGD([images], lr=1.)
            optim.zero_grad()
            optimx.zero_grad()
            output = self.forwardmetric(images)

            # calculate differentiable loss
            if iteration == 0:
                noise = 1e-7 * torch.randint_like(
                    output_orig, -1, 2, device=output_orig.device)
                # avoid zero gradient
                loss, _ = AdvRankLoss('ES', self.metric)(
                    output, output_orig + noise)
            else:
                loss, _ = AdvRankLoss('ES', self.metric)(
                    output, output_orig)
            itermsg = {'loss': loss.item()}
            loss.backward()

            # >> PGD: project SGD optimized result back to a valid region
            if self.pgditer > 1:
                images.grad.data.copy_(self.alpha * torch.sign(images.grad))
            elif self.pgditer == 1:
                images.grad.data.copy_(self.eps * torch.sign(images.grad))  # FGSM
            optimx.step()
            # L_infty constraint
            images = torch.min(images, images_orig + self.eps)
            # L_infty constraint
            images = torch.max(images, images_orig - self.eps)
            images = torch.clamp(images, min=0., max=1.)
            images = images.clone().detach()
            images.requires_grad = True

            # itermsg
            if int(os.getenv('PGD', -1)) > 0:
                print('(PGD)>', itermsg)

        # note: It's very critical to clear the junk gradients
        optim.zero_grad()
        optimx.zero_grad()
        images.requires_grad = False

        # evaluate adversarial samples
        xr = images.clone().detach()
        if self.verbose:
            r = images - images_orig
            with torch.no_grad():
                output = self.forwardmetric(images)
                # also calculate embedding shift
                if self.metric == 'C':
                    embshift = (1 - F.cosine_similarity(output, output_orig)
                                ).mean().item()
                elif self.metric in ('E', 'N'):
                    embshift = F.pairwise_distance(output, output_orig
                                                   ).mean().item()
        return xr

class QCSelector(object):
    '''
    Select C / Q for adversarial ranking attacks
    '''

    def __init__(self, attack_type: str, M: int = None,
                 W: int = None, SP: bool = False):
        self.attack_type = attack_type
        self.M = M
        self.W = W
        self.SP = SP
        # sensible choice due to SOP dataset property
        self.M_GT = 5  # [2002.11293 Locked. Don't modify!]
        self.map = {
            'ES': self._sel_es,
            'FOA': self._sel_foa,
            'CA+': self._sel_caplus,
            'CA-': self._sel_caminus,
            'QA+': self._sel_qaplus,
            'QA-': self._sel_qaminus,
            'GTM': self._sel_gtm,
            'GTT': self._sel_gtt,
            'TMA': self._sel_tma,
        }

    def __call__(self, *argv, **kwargs):
        with torch.no_grad():
            ret = self.map[self.attack_type](*argv, **kwargs)
        return ret

    def _sel_es(self, dist, candi):
        # -- [orig] untargeted attack, UT
        # >> the KNN accuracy, or say the Recall@1
        return None

    def _sel_foa(self, dist, candi):
        # == configuration for FOA:M=2, M>2
        M_GT = self.M_GT
        M = self.M

        # == select the M=2 candidates. note, x1 is closer to q than x2
        if self.M == 2:

            if True:
                # local sampling (default)
                topmost = int(candi[0].size(0) * 0.01)
                topxm = dist.topk(topmost + 1, dim=1,
                                  largest=False)[1][:, 1:]  # [output_0, M]
                sel = np.vstack([np.random.permutation(topmost)
                                 for j in range(topxm.shape[0])])
                msample = torch.stack([topxm[i][np.sort(sel[i, :M])]
                                    for i in range(topxm.shape[0])])
                if self.SP:
                    mgtruth = torch.stack([topxm[i][np.sort(sel[i, M:])[:M_GT]]
                                        for i in range(topxm.shape[0])])
            else:
                # global sampling
                distsort = dist.sort(dim=1)[1]  # [output_0, candi_0]
                mpairs = torch.randint(
                    candi[0].shape[0], (dist.shape[0], M)).sort(
                    dim=1)[0]  # [output_0, M]
                msample = torch.stack([distsort[i, mpairs[i]]
                                    for i in range(dist.shape[0])])  # [output_0, M]
                if self.SP:
                    # [output_0, M_GT]
                    mgtruth = dist.topk(
                        M_GT + 1, dim=1, largest=False)[1][:, 1:]
            embpairs = candi[0][msample, :]  # [output_0, M, output_1]
            if self.SP:
                embgts = candi[0][mgtruth, :]  # [output_0, M_GT, output_1]

        # == select M>2 candidates, in any order
        elif self.M > 2:

            if True:
                # local sampling (from topmost samples)
                topmost = int(candi[0].size(0) * 0.01)
                topxm = dist.topk(topmost + 1, dim=1,
                                  largest=False)[1][:, 1:]  # [output_0, M]
                sel = np.vstack([np.random.permutation(topmost)
                                 for j in range(topxm.shape[0])])
                msample = torch.stack([topxm[i][sel[i, :M]]
                                    for i in range(topxm.shape[0])])
                if self.SP:
                    mgtruth = torch.stack([topxm[i][np.sort(sel[i, M:])[:M_GT]]
                                        for i in range(topxm.shape[0])])
            else:
                # global sampling
                msample = torch.randint(
                    candi[0].shape[0], (dist.shape[0], M))  # [output_0, M]
                if self.SP:
                    mgtruth = dist.topk(
                        M_GT + 1, dim=1, largest=False)[1][:, 1:]
            embpairs = candi[0][msample, :]  # [output_0, M, output_1]
            if self.SP:
                embgts = candi[0][mgtruth, :]  # [output_0, M_GT, output_1]

        # return selections
        if self.SP:
            return (embpairs, msample, embgts, mgtruth)
        else:
            return (embpairs, msample)

    def _sel_caplus(self, dist, candi):
        # -- [orig] candidate attack, W=?
        # >> select W=? attacking targets
        '''
        return:
            msample : n x W(1)  n个随机的坐标
            embpairs: msample 对应的features
        '''
        if 'global' == os.getenv('SAMPLE', 'global'):
            msample = torch.randint(candi[0].shape[0], (dist.shape[0], self.W))  # [output_0, W]
        elif 'local' == os.getenv('SAMPLE', 'global'):
            local_lb = int(candi[0].shape[0] * 0.01)
            local_ub = int(candi[0].shape[0] * 0.05)
            topxm = dist.topk(local_ub + 1, dim=1, largest=False)[1][:, 1:]
            sel = np.random.randint(
                local_lb, local_ub, (dist.shape[0], self.W))
            msample = torch.stack([topxm[i][sel[i]]
                                for i in range(topxm.shape[0])])
        embpairs = candi[0][msample, :]
        return (embpairs, msample)

    def _sel_caminus(self, dist, candi):
        # these are not the extremely precise topW queries but an approximation
        # select W candidates from the topmost samples
        topmost = 10#int(candi[0].size(0) * 0.01)
        if int(os.getenv('VIS', 0)) > 0:
            topmost = int(candi[0].size(0) * 0.0003)
        topxm = dist.topk(topmost +
                          1, dim=1, largest=False)[1][:, 1:]  # [output_0, W]
        sel = np.random.randint(0, topmost, [topxm.shape[0], self.W])
        msample = torch.stack([topxm[i][sel[i]] for i in range(topxm.shape[0])])
        embpairs = candi[0][msample, :]  # [output_0, W, output_1]
        return (embpairs, msample)

    def _sel_qaplus(self, dist, candi):
        M = self.M
        # random sampling from populationfor QA+
        msample = torch.randint(candi[0].shape[0], (dist.shape[0], M))  # [output_0,M]

        embpairs = candi[0][msample, :]

        return (embpairs, msample)

    def _sel_qaminus(self, dist, candi) -> tuple:
        M = self.M
        # random sampling from top-3M for QA-
        topmost = 10    #int(candi[0].size(0) * 0.01)

        topxm = dist.topk(topmost + 1, dim=1, largest=False)[1][:, 1:]
        sel = np.vstack([np.random.permutation(topmost)
                         for i in range(dist.shape[0])])
        msample = torch.stack([topxm[i][sel[i, :M]]
                            for i in range(topxm.shape[0])])
        embpairs = candi[0][msample, :]

        return (embpairs, msample)

    def _sel_tma(self, dist, candi) -> tuple:
        # random sampling is enough. -- like QA+
        idxrand = torch.randint(candi[0].size(0), (dist.size(0),))
        embrand = candi[0][idxrand, :]
        return (embrand, idxrand)

    def _sel_gtm(self, dist, candi, loc_self) -> tuple:
        # top-1 matching and top-1 unmatching
        d = dist.clone()
        # filter out the query itself
        d[range(len(loc_self)), loc_self] = 1e38
        argsort = d.argsort(dim=1, descending=False)
        argrank = argsort.argsort(dim=1, descending=False)
        mylabel = candi[1][argsort[:, 0]].view(-1, 1)
        mask_match = (candi[1].view(1, -1) == mylabel)
        mask_unmatch = (candi[1].view(1, -1) != mylabel)
        fstmatch = torch.stack([argrank[i, mask_match[i]].argmin()
                             for i in range(mask_match.size(0))])
        fstunmatch = torch.stack([argrank[i, mask_unmatch[i]].argmin()
                               for i in range(mask_unmatch.size(0))])
        ret = ((candi[0][fstmatch, None, :], fstmatch),
               (candi[0][fstunmatch, None, :], fstunmatch),
               (candi[0][loc_self, None, :], loc_self))
        return ret

    def _sel_gtt(self, dist, candi, loc_self) -> tuple:
        # non-self top-2
        d = dist.clone()
        d[range(len(loc_self)), loc_self] = 1e38
        idtop2 = d.topk(2, dim=-1, largest=False)[1]
        fstidm = idtop2[:, 0]
        fstidum = idtop2[:, 1]
        ret = ((candi[0][fstidm, None, :], fstidm),
               (candi[0][fstidum, None, :], fstidum),
               (candi[0][loc_self, None, :], loc_self))
        return ret

class AdvRankLoss(object):
    '''
    Factory of all types of loss functions used in ranking attacks
    '''

    def RankLossEmbShift(self, repv: torch.Tensor, repv_orig: torch.Tensor):
        '''
        Computes the embedding shift, we want to maximize it by gradient descent

        Arguments:
            repv: size(batch, embdding_dim), requires_grad.
            repv_orig: size(batch, embedding_dim).
        '''
        if self.metric == 'C':
            distance = 1 - F.cosine_similarity(repv, repv_orig)
            # distance = -(1 - torch.mm(repv, repv_orig)).trace() # not efficient
        elif self.metric in ('E', 'N'):
            distance = F.pairwise_distance(repv, repv_orig)
        loss = -distance.sum()
        return (loss, None)

    def RankLossQueryAttack(self, qs: torch.Tensor, Cs: torch.Tensor, Xs: torch.Tensor,
                            *, pm: str, dist: torch.Tensor = None, cidx: torch.Tensor = None):
        '''
        Computes the loss function for pure query attack

        Arguments:
            qs: size(batch, embedding_dim), query embeddings.
            Cs: size(batch, M, embedding_dim), selected candidates.
            Xs: size(testsize, embedding_dim), embedding of test set.
            pm: either '+' or '-'.
            dist: size(batch, testsize), pairwise distance matrix.
            cidx: size(batch, M), index of candidates in Xs.
        '''
        assert(qs.shape[1] == Cs.shape[2] == Xs.shape[1])
        NIter, M, D, NX = qs.shape[0], Cs.shape[1], Cs.shape[2], Xs.shape[0]
        DO_RANK = (dist is not None) and (cidx is not None)
        losses, ranks = [], []
        #refrank = []
        for i in range(NIter):
            # == compute the pairwise loss
            q = qs[i].view(1, D)  # [1, output_1]
            C = Cs[i, :, :].view(M, D)  # [1, output_1]
            if self.metric == 'C':
                A = (1 - torch.mm(q, C.t())).view(1, M)
                B = (1 - torch.mm(Xs, q.t())).view(NX, 1)
            elif self.metric in ('E', 'N'):
                A = torch.cdist(q, C).view(1, M)
                B = torch.cdist(Xs, q).view(NX, 1)
                # [XXX] the old method suffer from large memory footprint
                # A = (C - q).norm(2, dim=1).view(1, M)
                # B = (Xs - q).norm(2, dim=1).view(NX, 1)
            # == loss function
            if '+' == pm:
                loss = (A - B).clamp(min=0.).mean()
            elif '-' == pm:
                loss = (-A + B).clamp(min=0.).mean()
            losses.append(loss)
            # == compute the rank
            if DO_RANK:
                ranks.append(torch.mean(dist[i].flatten().argsort().argsort()
                                     [cidx[i, :].flatten()].float()).item())
            #refrank.append( ((A>B).float().mean()).item() )
        #print('(debug)', 'rank=', statistics.mean(refrank))
        loss = torch.stack(losses).mean()
        rank = statistics.mean(ranks) if DO_RANK else None
        return (loss, rank)

    def RankLossQueryAttackDistance(self, qs: torch.Tensor, Cs: torch.Tensor, Xs: torch.Tensor, *,
                                    pm: str, dist: torch.Tensor = None, cidx: torch.Tensor = None):
        '''
        the distance based objective is worse.
        '''
        assert(qs.shape[1] == Cs.shape[2] == Xs.shape[1])  # D
        N, M, D, NX = qs.shape[0], Cs.shape[1], Cs.shape[2], Xs.shape[0]
        DO_RANK = (dist is not None) and (cidx is not None)
        losses, ranks = [], []
        for i in range(N):
            q = qs[i].view(1, D)
            C = Cs[i, :, :].view(M, D)
            if (self.metric, pm) == ('C', '+'):
                loss = (1 - torch.mm(q, C.t())).mean()
            elif (self.metric, pm) == ('C', '-'):
                loss = -(1 - torch.mm(q, C.t())).mean()
            elif (self.metric, pm) == ('E', '+'):
                loss = (C - q).norm(2, dim=1).mean()
            elif (self.metric, pm) == ('E', '-'):
                loss = -(C - q).norm(2, dim=1).mean()
            elif (self.metric, pm) == ('N', '+'):
                loss = (C - q).norm(2, dim=1).mean()
            elif (self.metric, pm) == ('N', '-'):
                loss = -(C - q).norm(2, dim=1).mean()
            losses.append(loss)
            if DO_RANK:
                if self.metric == 'C':
                    A = (1 - torch.mm(q, C.t())).expand(NX, M)
                    B = (1 - torch.mm(Xs, q.t())).expand(NX, M)
                elif self.metric in ('E', 'N'):
                    A = (C - q).norm(2, dim=1).expand(NX, M)
                    B = (Xs - q).norm(2, dim=1).view(NX, 1).expand(NX, M)
                # non-normalized result
                rank = ((A > B).float().mean() * NX).item()
                ranks.append(rank)
        loss = torch.stack(losses).mean()
        rank = statistics.mean(ranks) if DO_RANK else None
        return (loss, rank)

    def RankLossCandidateAttack(
            self, cs: torch.Tensor, Qs: torch.Tensor, Xs: torch.Tensor, *, pm: str):
        '''
        Computes the loss function for pure candidate attack

        Arguments:
            cs: size(batch, embedding_dim), embeddings of candidates.
            Qs: size(batch, W, embedding_dim), embedding of selected queries.
            Xs: size(testsize, embedding_dim), embedding of test set.
            pm: either '+' or '-'
        '''
        assert(cs.shape[1] == Qs.shape[2] == Xs.shape[1])
        NIter, W, D, NX = cs.shape[0], Qs.shape[1], Qs.shape[2], Xs.shape[0]
        losses, ranks = [], []
        for i in range(NIter):
            # == compute pairwise distance
            c = cs[i].view(1, D)  # [1, output_1]
            Q = Qs[i, :, :].view(W, D)  # [W, output_1]
            if self.metric == 'C':
                A = 1 - torch.mm(c, Q.t()).expand(NX, W)  # [candi_0, W]
                B = 1 - torch.mm(Xs, Q.t())  # [candi_0, W]
            elif self.metric in ('E', 'N'):
                A = (Q - c).norm(2, dim=1).expand(NX, W)  # [candi_0, W]
                B = torch.cdist(Xs, Q, p=2.0)
                # B2 = (Xs.view(NX, 1, D).expand(NX, W, D) -
                #     Q.view(1, W, D).expand(NX, W, D)).norm(2, dim=2)  # [candi_0, W]
                #assert((B-B2).abs().norm() < 1e-4)
            # == loss function
            if '+' == pm:
                loss = (A - B).clamp(min=0.).mean()
            elif '-' == pm:
                loss = (-A + B).clamp(min=0.).mean()
            losses.append(loss)
            # == compute the rank. Note, the > sign is correct
            rank = ((A > B).float().mean() * NX).item()
            ranks.append(rank)
        loss = torch.stack(losses).mean()
        rank = statistics.mean(ranks)
        return (loss, rank)

    def RankLossCandidateAttackDistance(
            self, cs: torch.Tensor, Qs: torch.Tensor, Xs: torch.Tensor, *, pm: str):
        '''
        Computes the loss function for pure candidate attack
        using the inferior distance objective
        '''
        assert(cs.shape[1] == Qs.shape[2] == Xs.shape[1])
        NIter, W, D, NX = cs.shape[0], Qs.shape[1], Qs.shape[2], Xs.shape[0]
        losses, ranks = [], []
        for i in range(NIter):
            c = cs[i].view(1, D)
            Q = Qs[i, :, :].view(W, D)
            if (self.metric, pm) == ('C', '+'):
                loss = (1 - torch.mm(c, Q.t())).mean()
            elif (self.metric, pm) == ('C', '-'):
                loss = -(1 - torch.mm(c, Q.t())).mean()
            elif (self.metric, pm) == ('E', '+'):
                loss = (Q - c).norm(2, dim=1).mean()
            elif (self.metric, pm) == ('E', '-'):
                loss = -(Q - c).norm(2, dim=1).mean()
            elif (self.metric, pm) == ('N', '+'):
                loss = (Q - c).norm(2, dim=1).mean()
            elif (self.metric, pm) == ('N', '-'):
                loss = -(Q - c).norm(2, dim=1).mean()
            losses.append(loss)
            if self.metric == 'C':
                A = (1 - torch.mm(c, Q.t())).expand(NX, W)
                B = (1 - torch.mm(Xs, Q.t()))
            elif self.metric in ('E', 'N'):
                A = (Q - c).norm(2, dim=1).expand(NX, W)
                B = (Xs.view(NX, 1, D).expand(NX, W, D) -
                     Q.view(1, W, D).expand(NX, W, D)).norm(2, dim=2)
            rank = ((A > B).float().mean() * NX).item()  # ">" sign is correct
            ranks.append(rank)
        loss = torch.stack(losses).mean()
        rank = statistics.mean(ranks)
        return (loss, rank)

    def RankLossFullOrderM2Attack(
            self, qs: torch.Tensor, ps: torch.Tensor, ns: torch.Tensor):
        '''
        Computes the loss function for M=2 full-order attack

        Arguments:
            qs: size(batch, embedding_dim), queries/anchors
            ps: size(batch, embedding_dim), positive samples
            ns: size(batch, embedding_dim), negative samples
        '''
        assert(qs.shape[0] == ps.shape[0] == ns.shape[0])
        assert(qs.shape[1] == ps.shape[1] == ns.shape[1])
        if self.metric == 'C':
            dist1 = 1 - torch.nn.functional.cosine_similarity(qs, ps, dim=1)
            dist2 = 1 - torch.nn.functional.cosine_similarity(qs, ns, dim=1)
        elif self.metric in ('E', 'N'):
            dist1 = torch.nn.functional.pairwise_distance(qs, ps, p=2)
            dist2 = torch.nn.functional.pairwise_distance(qs, ns, p=2)
        else:
            raise ValueError(self.metric)
        loss = (dist1 - dist2).clamp(min=0.).mean()
        acc = (dist1 <= dist2).sum().item() / qs.shape[0]
        return (loss, acc)

    def RankLossFullOrderMXAttack(self, qs: torch.Tensor, Cs: torch.Tensor):
        assert(qs.shape[1] == Cs.shape[2])
        NIter, M, D = qs.shape[0], Cs.shape[1], Cs.shape[2]
        losses, taus = [], []
        for i in range(NIter):
            q = qs[i].view(1, D)
            C = Cs[i, :, :].view(M, D)
            if self.metric == 'C':
                dist = 1 - torch.mm(q, C.t())
            elif self.metric in ('E', 'N'):
                dist = (C - q).norm(2, dim=1)
            tau = stats.kendalltau(
                np.arange(M), dist.cpu().detach().numpy())[0]
            taus.append(tau)
            dist = dist.expand(M, M)
            loss = (dist.t() - dist).triu(diagonal=1).clamp(min=0.).mean()
            losses.append(loss)
        loss = torch.stack(losses).mean()
        tau = statistics.mean(taus)
        return (loss, tau)

    def RankLossGreedyTop1Misrank(self, qs: torch.Tensor, emm: torch.Tensor,
                                  emu: torch.Tensor, ems: torch.Tensor, Xs: torch.Tensor):
        '''
        <Compound loss> Greedy Top-1 Misranking. (GTM)
        Goal: top1.class neq original class.
        Arguments:
            qs: size(batch, embedding_dim), query to be perturbed.
            emm: size(batch, 1, embdding_dim), top-1 matching candidate
            emu: size(batch, 1, embdding_dim), top-1 unmatching cancidate

        Observations[MNIST-rc2f2-ptripletN]:
            1) only loss_match: very weak compared to ES
            2) only loss_unmatch: relatively weak
            3) combined: weak
            4) loss_unmatch (dist)
            5) loss_match (dist)
            6) dist both
            7) dist match
        [*] 8) dist unmatch (still best after qc selection bugfix)
        '''
        assert(qs.shape[1] == emm.shape[2] == emu.shape[2])
        #loss_match, _ = self.funcmap['QA-'](qs, emm, Xs)
        #loss_match, _ = self.funcmap['QA-DIST'](qs, emm, Xs, pm='-')
        #loss_unmatch, _ = self.funcmap['QA+'](qs, emu, Xs)
        #loss_unmatch, _ = self.funcmap['QA-DIST'](qs, emu, Xs, pm='+')
        #loss = loss_match + loss_unmatch
        # [scratch]
        emm = emm.squeeze()
        emu = emu.squeeze()
        ems = ems.squeeze()
        if self.metric in ('C',):
            #l_m = -(1 - F.cosine_similarity(qs, emm))
            l_u = (1 - F.cosine_similarity(qs, emu))
        elif self.metric in ('E', 'N'):
            #l_m = -F.pairwise_distance(qs, emm)
            l_u = F.pairwise_distance(qs, emu)
            #l_s = -F.pairwise_distance(qs, ems)
        loss = (l_u).mean()
        return loss

    def RankLossGreedyTop1Translocation(self, qs: torch.Tensor, emm: torch.Tensor,
                                        emu: torch.Tensor, ems: torch.Tensor, Xs: torch.Tensor):
        '''
        <Compound loss> Greedy Top-1 Translocation (GTT)
        Goal: top1.identity neq original identity.
        Arguments:
            see document for GTM
        observations:
            1) TODO
        '''
        assert(qs.shape[1] == emm.shape[2] == emu.shape[2])
        loss_match, _ = self.funcmap['QA-'](qs, emm, Xs)
        #loss_match, _ = self.funcmap['QA-DIST'](qs, emm, Xs, pm='-')
        #loss_unmatch, _ = self.funcmap['QA+'](qs, emu, Xs)
        #loss_unmatch, _ = self.funcmap['QA-DIST'](qs, emu, Xs, pm='+')
        loss = loss_match  # + loss_unmatch
        # [scratch]
        #emm = emm.squeeze()
        #emu = emu.squeeze()
        #ems = ems.squeeze()
        # if self.metric in ('C',):
        #    #l_m = -(1 - F.cosine_similarity(qs, emm))
        #    l_u = (1 - F.cosine_similarity(qs, emu))
        # elif self.metric in ('E', 'N'):
        #    #l_m = -F.pairwise_distance(qs, emm)
        #    #l_u = F.pairwise_distance(qs, emu)
        #    l_s = -F.pairwise_distance(qs, ems)
        #loss = (l_s).mean()
        return loss

    def RankLossTargetedMismatchAttack(
            self, qs: torch.Tensor, embrand: torch.Tensor):
        '''
        Targeted Mismatch Attack using Global Descriptor (ICCV'19)
        https://arxiv.org/pdf/1908.09163.pdf
        '''
        assert(qs.shape[0] == embrand.shape[0])
        #assert(self.metric in ('C', 'N'))
        loss = (1 - F.cosine_similarity(qs, embrand)).mean()
        return loss

    def RankLossLearningToMisrank(self, qs: torch.Tensor, embp: torch.Tensor,
                                  embn: torch.Tensor):
        '''
        Learning-To-Mis-Rank
        But the paper did not specify a margin.
        Following Eq.1 of https://arxiv.org/pdf/2004.04199.pdf
        '''
        assert(qs.shape == embp.shape == embn.shape)
        if self.metric == 'C':
            loss = (1 - F.cosine_similarity(qs, embn)) - \
                   (1 - F.cosine_similarity(qs, embp))
        elif self.metric in ('N', 'E'):
            loss = F.pairwise_distance(qs, embn) - \
                F.pairwise_distance(qs, embp)
        return loss.mean()

    def __init__(self, request: str, metric: str):
        '''
        Initialize various loss functions
        '''
        assert(metric in ('E', 'N', 'C'))
        self.metric = metric
        self.funcmap = {
            'ES': self.RankLossEmbShift,
            'QA': self.RankLossQueryAttack,
            'QA+': functools.partial(self.RankLossQueryAttack, pm='+'),
            'QA-': functools.partial(self.RankLossQueryAttack, pm='-'),
            'QA-DIST': self.RankLossQueryAttackDistance,
            'CA': self.RankLossCandidateAttack,
            'CA+': functools.partial(self.RankLossCandidateAttack, pm='+'),
            'CA-': functools.partial(self.RankLossCandidateAttack, pm='-'),
            'CA-DIST': self.RankLossCandidateAttackDistance,
            'FOA2': self.RankLossFullOrderM2Attack,
            'FOAX': self.RankLossFullOrderMXAttack,
            'GTM': self.RankLossGreedyTop1Misrank,
            'GTT': self.RankLossGreedyTop1Translocation,
            'TMA': self.RankLossTargetedMismatchAttack,
            'LTM': self.RankLossLearningToMisrank,
        }
        if request not in self.funcmap.keys():
            raise KeyError(f'Requested loss function "{request}" not found!')
        self.request = request

    def __call__(self, *args, **kwargs):
        '''
        Note, you should handle the normalization outside of this class.
        The input and output of the function also vary based on the request.
        '''
        return self.funcmap[self.request](*args, **kwargs)
