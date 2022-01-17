


import functools as ft
import torch.nn.functional as F
import torch
import numpy as np
from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.reid_patch import eval_ssim, get_result, get_train_set, make_dict
from .robrank import *
import time
margin_cosine: float = 0.2
margin_euclidean: float = 1.0
device = 'cuda'

class robrank_defense:
    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self.train_set = get_train_set(self.cfg)

    def get_defense_result(self):
        return get_result(self.cfg,self.cfg.MODEL.DEFENSE_TRAINED_WEIGHT,'defense')

    def defense(self):
        RobRank_defense(self.cfg,self.train_set,self.cfg.DEFENSEMETHOD)

def RobRank_defense(cfg,train_set,str):
    if str =='SES':
        train_step = ses_training_step
    elif str == 'EST':
        train_step = est_training_step
    elif str == 'PNP':
        train_step = ACT_training_step
    else:
        raise KeyError

    cfg = DefaultTrainer.auto_scale_hyperparams(cfg,train_set.dataset.num_classes)
    model = DefaultTrainer.build_model_main(cfg)#this model was used for later evaluations
    model.RESIZE = True
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)

    optimizer = DefaultTrainer.build_optimizer(cfg,model)
    max_id =4000
    EPOCH = 3
    #eval_train(model,train_set,max_id=500)
    
    for epoch in range(EPOCH):
        model.train()
        loss_total = 0
        print(f'start training for epoch {epoch} of {EPOCH}')
        time_stamp_start = time.strftime("%H:%M:%S", time.localtime()) 
        for id,data in enumerate(train_set):
            if id>max_id:
                break
            
            images = (data['images']/255.0).to(device)
            labels = data['targets'].to(device)

            #model.heads.MODE = 'F'
            optimizer.zero_grad()
            loss = train_step(model,images,labels,id)

            loss_total+=loss
            loss.backward()
            optimizer.step()
        #eval_train(model,train_set,max_id=500)
        time_stamp_end = time.strftime("%H:%M:%S", time.localtime()) 
        print(f'total_loss for epoch {epoch} of {EPOCH} is {loss_total} | {time_stamp_start} - {time_stamp_end}')
    
    print(f'finished {str}_training !')
    Checkpointer(model,'model').save(f'{str}_{cfg.DATASETS.NAMES[0]}_{cfg.CFGTYPE}')
    print(f'Successfully saved the {str}_trained model !')

def est_training_step(model, images, labels,id):
    '''
    Do adversarial training using Mo's defensive triplet (2002.11293 / ECCV'20)
    Embedding-Shifted Triplet (EST)
    Confirmed for MNIST/Fashion-MNIST/CUB/CARS/SOP
    This defense is generic to network architecture and metric learning loss.

    Arguments:
        model: a pytorch_lightning.LightningModule instance. It projects
               input images into a embedding space.
        batch: see pytorch lightning protocol for training_step(...)
        batch_idx: see pytorch lightning protocol for training_step(...)
    '''
    # generate adversarial examples
    #advatk_metric = 'C' if model.dataset in ('mnist', 'fashion') else 'C'
    model.heads.MODE = 'F'
    advrank = AdvRank(model, eps=4. / 255.,
                      alpha=1. / 255.,
                      pgditer=24,
                      device = 'cuda',
                      metric='N', verbose=False)
    # set shape

    # generate adv examples
    advimgs = advrank.embShift(images)
    if id%800==0:
        print('ssim = ',eval_ssim(images,advimgs))
                
    model.heads.MODE = 'FC'
    outputs = model(make_dict(advimgs,labels))

    loss_dict = model.losses(outputs,labels)
    loss = sum(loss_dict.values())

    return loss

# def svd(output,constraints: list = [6, 7]):
#     Constraints = {
#     1: lambda svd: (svd.S.max() - 1e+3).relu(),
#     2: lambda svd: (svd.S[0] / svd.S[1] - 8.0).relu(),
#     3: lambda svd: (1.2 - svd.S[1:5] / svd.S[2:6]).relu().mean(),
#     4: lambda svd: (1e-3 / svd.S.min() - 1.0).relu(),
#     5: lambda svd: torch.exp(svd.S.max() / (8192 * svd.S.min()) - 1),
#     6: lambda svd: torch.log(svd.S.max() / 2e+2).relu(),
#     7: lambda svd: torch.log(1e-5 / svd.S.min()).relu(),
#     8: lambda svd: torch.log(svd.S.max() / (1e5 * svd.S.min())).relu()
#     }
#     svd = torch.svd(output)
#     penalties = [Constraints[i](svd) for i in constraints]
#     return ft.reduce(op.add, penalties) if penalties else 0.0

def mmt_training_step(model: torch.nn.Module, images,labels,id):
    '''
    min-max triplet
    '''
    # evaluate original benign sample
    model.eval()
    model.heads.MODE = 'C'
    loss_fun = ptriplet()
    with torch.no_grad():
        output_orig = model(images)
    # generate adversarial examples
    triplets = miner(output_orig, labels, method=loss_fun._minermethod,
                     metric=loss_fun._metric,
                     margin=margin_euclidean if loss_fun._metric in ('E', 'N')
                     else margin_cosine)
    model.eval()
    pnp = PositiveNegativePerplexing(model, eps=4. / 255.,
                      alpha=1. / 255.,
                      pgditer=24,
                      device = 'cuda',
                      metric='N', verbose=False)
    model.eval()
    images_pnp = pnp.minmaxtriplet(images, triplets)
    
    if id % 800 == 0:
        print(f'ssim = {eval_ssim(images,images_pnp)}')

    pnemb = model(images_pnp)
    if loss_fun._metric in ('C', 'N'):
        pnemb = F.normalize(pnemb)
    # compute adversarial loss
    
    loss = loss_fun.raw(
        pnemb[:len(pnemb) // 3],
        pnemb[len(pnemb) // 3:2 * len(pnemb) // 3],
        pnemb[2 * len(pnemb) // 3:]).mean()
    
                
    # model.heads.MODE = 'FC'
    # outputs = model(adv_images)

    # #outputs = model(images)

    # loss_dict = model.losses(outputs,labels)
    # loss = sum(loss_dict.values())

    return loss

def ACT_training_step(model: torch.nn.Module, images,labels,id):
    '''
    Adversarial training with Positive/Negative Perplexing (PNP) Attack.
    Function signature follows pytorch_lightning.LightningModule.training_step,
    where model is a lightning model, batch is a tuple consisting images
    (torch.Tensor) and labels (torch.Tensor), and batch_idx is just an integer.

    Collapsing positive and negative -- Anti-Collapse (ACO) defense.
    force the model to learning robust feature and prevent the
    adversary from exploiting the non-robust feature and collapsing
    the positive/negative samples again. This is exactly the ACT defense
    discussed in https://arxiv.org/abs/2106.03614

    This defense is not agnostic to backbone architecure and metric learning
    loss. But it is recommended to use it in conjunction with triplet loss.
    '''
    model.eval()
    model.heads.MODE = 'F'
    with torch.no_grad():
        output_orig = model(images)

    model.train()
    # generate adversarial examples
    loss_fun = ptriplet()
    triplets = miner(output_orig, labels, method=loss_fun._minermethod,
                     metric=loss_fun._metric,
                     margin=margin_euclidean if loss_fun._metric in ('E', 'N')
                     else margin_cosine)
    anc, pos, neg = triplets
    pnp = PositiveNegativePerplexing(model, eps=4. / 255.,
                                    alpha=1. / 255.,
                                    pgditer=24,
                                    device = 'cuda',
                                    metric='N', verbose=False)
    # Collapsing positive and negative -- Anti-Collapse Triplet (ACT) defense.
    images_pnp = pnp.pncollapse(images, triplets)

    # Adversarial Training
    model.train()
    model.heads.MODE = 'C'
    pnemb = model(make_dict(images_pnp,labels))
    aemb = model(make_dict(images[anc, :, :, :],labels))
    pnemb = F.normalize(pnemb)
    aemb = F.normalize(aemb)
    # compute adversarial loss
    loss = loss_fun.raw(aemb, pnemb[:len(pnemb) // 2],
                              pnemb[len(pnemb) // 2:]).mean()

    return loss

def ses_training_step(model, images,labels,id):
    '''
    Adversarial training by directly supressing embedding shift (SES)
    max(*.es)->advimg, min(advimg->emb,oriimg->img;*.metric)
    Confirmed for MNIST/Fashion-MNIST
    [ ] for CUB/SOP

    This defense has been discussed in the supplementary material / appendix
    of the ECCV20 paper. (See arxiv: 2002.11293)
    '''
    model.heads.MODE = 'F'
    # images = images.clone().detach()
    # images.requires_grad_()
        # generate adversarial examples
    advrank = AdvRank(model, eps=4. / 255.,
                      alpha=1. / 255.,
                      pgditer=24,
                      device = 'cuda',
                      metric='N', verbose=False)
    advimgs = advrank.embShift(images)

    if id % 800 ==0:
        print(f'ssim = {eval_ssim(images,advimgs)}')
    # evaluate advtrain loss
    model.heads.MODE ='FC'
    output_orig = model(make_dict(advimgs,labels))
    loss_orig = model.losses(output_orig, labels)
    model.heads.MODE = 'F'
    output_adv = model(advimgs)
    # select defense method
    output_orig = output_orig['features']
    nori = F.normalize(output_orig)
    nadv = F.normalize(output_adv)
    embshift = F.pairwise_distance(nadv, nori)
    # loss and log
    # method 1: loss_triplet + loss_embshift
    loss = sum(loss_orig.values()) + 1.0 * embshift.mean()
    # method 2: loss_triplet + loss_embshiftp2
    #loss = loss_orig + 1.0 * (embshift ** 2).mean()

    return loss

class ptriplet(torch.nn.Module):
    _datasetspec = 'SPC-2'
    _minermethod = 'spc2-random'
    _xa = False
    def __init__(self):
        super().__init__()
        self._metric ='N'

    def __call__(self, *args, **kwargs):
        return ft.partial(fn_ptriplet,
                              metric=self._metric,
                              minermethod=self._minermethod,
                              xa=self._xa)(*args, **kwargs)


    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec

    def raw(self, repA, repP, repN, *, override_margin: float = None):
        margin_cosine: float = 0.2
        margin_euclidean: float = 1.0
        if self._metric in ('C', 'N'):
            margin =margin_cosine
        loss = fn_ptriplet_kernel(repA, repP, repN,
                                  metric=self._metric, margin=margin)
        return loss

def fn_ptriplet(repres: torch.Tensor, labels: torch.Tensor,
                *, metric: str, minermethod: str, p_switch: float = -1.0, xa: bool = False):
    '''
    Variant of triplet loss that accetps [cls=1,cls=1,cls=2,cls=2] batch.
    This corresponds to the SPC-2 setting in the ICML20 paper.

    metrics: C = cosine, E = euclidean, N = normalization + euclidean
    '''
    # Determine the margin for the specific metric
    if metric in ('C', 'N'):
        margin = margin_cosine
        repres = F.normalize(repres, p=2, dim=-1)
    elif metric in ('E',):
        margin = margin_euclidean
    # Sample the triplets
    anc, pos, neg = miner(repres, labels, method=minermethod,
                          metric=metric, margin=margin, p_switch=p_switch)
    if xa:
        return fn_ptriplet_kernel(repres[anc, :].detach(), repres[pos, :],
                                  repres[neg, :], metric=metric, margin=margin)
    loss = fn_ptriplet_kernel(repres[anc, :], repres[pos, :], repres[neg, :],
                              metric=metric, margin=margin)
    return loss

def fn_ptriplet_kernel(repA: torch.Tensor, repP: torch.Tensor, repN: torch.Tensor,
                       *, metric: str, margin: float):
    '''
    <functional> the core computation for spc-2 triplet loss.
    '''
    if metric == 'C':
        dap = 1 - F.cosine_similarity(repA, repP, dim=-1)
        dan = 1 - F.cosine_similarity(repA, repN, dim=-1)
        loss = (dap - dan + margin).relu().mean()
    elif metric in ('E', 'N'):
        #dap = F.pairwise_distance(repA, repP, p=2)
        #dan = F.pairwise_distance(repA, repN, p=2)
        #loss = (dap - dan + margin).relu().mean()
        loss = F.triplet_margin_loss(repA, repP, repN, margin=margin)
    else:
        raise ValueError(f'Illegal metric type {metric}!')
    return loss

def miner(repres: torch.Tensor, labels: torch.Tensor, *,
          method: str = 'random-triplet', metric: str = None,
          margin: float = None, p_switch: float = -1.0):
    '''
    Dispatcher for different batch data miners
    '''
    assert(len(repres.shape) == 2)
    assert(metric is not None)
    labels = labels.view(-1)
    anchor, positive, negative = __miner_spc2_random(repres, labels)
    
    if p_switch > 0.0 and (np.random.rand() < p_switch):
        # spectrum regularization (following ICML20 paper text description)
        # return (anchor, negative, positive)  # XXX: lead to notable performance drop
        # spectrum regulairzation (following upstream code)
        return (anchor, anchor, positive)
    return (anchor, positive, negative)

def __miner_spc2_random(repres: torch.Tensor, labels: torch.Tensor): 
    '''
    Sampling triplets from pairwise data
    '''
    import random
    negs = []
    for i in range(labels.nelement() // 2):
        # [ method 1: 40it/s legion
        mask_neg = (labels != labels[2 * i])
        if mask_neg.sum() > 0:
            negs.append(random.choice(torch.where(mask_neg)[0]).item())
        else:
            # handle rare/corner cases where the batch is bad
            negs.append(np.random.choice(len(labels)))
        # [ method 2: 35it/s legion
        # candidates = tuple(filter(lambda x: x // 2 != i,
        #                          range(labels.nelement())))
        # while True:
        #    neg = random.sample(candidates, 1)
        #    if labels[i * 2].item() != labels[neg].item():
        #        break
        # negs.append(*neg)
    anchors = torch.arange(0, len(labels), 2)
    positives = torch.arange(1, len(labels), 2)
    negatives = torch.tensor(negs, dtype=torch.long, device=repres.device)
    return (anchors, positives, negatives)

class PositiveNegativePerplexing(object):

    '''
    Attack designed for adversarial training
    '''

    def __init__(self,
                 model: torch.nn.Module, eps: float, alpha: float, pgditer: int,
                 device: str, metric: str, verbose: bool = False):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.pgditer = pgditer
        self.device = device
        self.metric = metric
        self.verbose = verbose

    def pncollapse(self, images: torch.Tensor, triplets: tuple):
        '''
        collapse the positive and negative sample in the embedding space.
        (p->, <-n)
        '''
        # prepare
        anc, pos, neg = triplets
        impos = images[pos, :, :, :].clone().detach().to(self.device)
        imneg = images[neg, :, :, :].clone().detach().to(self.device)
        images_orig = torch.cat([impos, imneg]).clone().detach()
        images = torch.cat([impos, imneg])
        images.requires_grad = True
        # start PGD
        self.model.eval()
        for iteration in range(self.pgditer):
            # optimizer
            optm = torch.optim.SGD(self.model.parameters(), lr=0.)
            optx = torch.optim.SGD([images], lr=1.)
            optm.zero_grad()
            optx.zero_grad()
            # forward data to be perturbed
            emb = self.model(images)
            if self.metric in ('C', 'N'):
                emb = F.normalize(emb)
            # draw two samples close to each other
            if self.metric in ('E', 'N'):
                loss = F.pairwise_distance(emb[:len(emb) // 2],
                                           emb[len(emb) // 2:]).mean()
            elif self.metric in ('C',):
                loss = 1 - F.cosine_similarity(emb[:len(emb) // 2],
                                               emb[len(emb) // 2:]).mean()
            itermsg = {'loss': loss.item()}
            loss.backward()
            # projected gradient descent
            if self.pgditer > 1:
                images.grad.data.copy_(self.alpha * torch.sign(images.grad))
            elif self.pgditer == 1:
                images.grad.data.copy_(self.eps * torch.sign(images.grad))
            optx.step()
            images = torch.min(images, images_orig + self.eps)
            images = torch.max(images, images_orig - self.eps)
            images = torch.clamp(images, min=0., max=1.)
            images = images.clone().detach()
            images.requires_grad = True
            # report for the current iteration
            if self.verbose:
                print(f'(PGD)>', itermsg)
        # note: it is very important to clear the junk gradients.
        optm.zero_grad()
        optx.zero_grad()
        images.requires_grad = False
        # return the adversarial example
        if self.verbose:
            print(images.shape)
        # images: concatenation of adversarial positive and negative
        return images

    def pnanchor(self, images: torch.Tensor, triplets: tuple,
                 emb_anchor_detached: torch.Tensor):
        '''
        (a, p->), (a, <-n), adversary to contrastive
        '''
        # prepare
        anc, pos, neg = triplets
        impos = images[pos, :, :, :].clone().detach().to(self.device)
        imneg = images[neg, :, :, :].clone().detach().to(self.device)
        images_orig = torch.cat([impos, imneg]).clone().detach()
        images = torch.cat([impos, imneg])
        images.requires_grad = True
        # start PGD
        self.model.eval()
        for iteration in range(self.pgditer):
            # optimizer
            optm = torch.optim.SGD(self.model.parameters(), lr=0.)
            optx = torch.optim.SGD([images], lr=1.)
            optm.zero_grad()
            optx.zero_grad()
            # forward data to be perturbed
            emb = self.model.forward(images)
            if self.metric in ('C', 'N'):
                emb = F.normalize(emb)
            # (a, p->), (a, <-n)
            if self.metric in ('E', 'N'):
                loss = F.pairwise_distance(emb_anchor_detached,
                                           emb[len(emb) // 2:])
                loss -= F.pairwise_distance(emb_anchor_detached,
                                            emb[:len(emb) // 2])
                loss = loss.mean()
            elif self.metric in ('C',):
                loss = 1 - F.cosine_similarity(emb_anchor_detached,
                                               emb[len(emb) // 2:])
                loss -= 1 - F.cosine_similarity(emb_anchor_detached,
                                                emb[:len(emb) // 2])
                loss = loss.mean()
            itermsg = {'loss': loss.item()}
            loss.backward()
            # projected gradient descent
            if self.pgditer > 1:
                images.grad.data.copy_(self.alpha * torch.sign(images.grad))
            elif self.pgditer == 1:
                images.grad.data.copy_(self.eps * torch.sign(images.grad))
            optx.step()
            images = torch.min(images, images_orig + self.eps)
            images = torch.max(images, images_orig - self.eps)
            images = torch.clamp(images, min=0., max=1.)
            images = images.clone().detach()
            images.requires_grad = True
            # report for the current iteration
            if self.verbose:
                print(f'(PGD)>', itermsg)
        # note: it is very important to clear the junk gradients.
        optm.zero_grad()
        optx.zero_grad()
        images.requires_grad = False
        # return the adversarial example
        if self.verbose:
            print(images.shape)
        return images

    def apsplit(self, images: torch.Tensor, triplets: tuple):
        '''
        maximize d(a, p)
        '''
        # prepare
        anc, pos, neg = triplets
        imanc = images[anc, :, :, :].clone().detach().to(self.device)
        impos = images[pos, :, :, :].clone().detach().to(self.device)
        images_orig = torch.cat([imanc, impos]).clone().detach()
        images = torch.cat([imanc, impos])
        images.requires_grad = True
        # start PGD
        self.model.eval()
        for iteration in range(self.pgditer):
            # optimizer
            optm = torch.optim.SGD(self.model.parameters(), lr=0.)
            optx = torch.optim.SGD([images], lr=1.)
            optm.zero_grad()
            optx.zero_grad()
            # forward data to be perturbed
            emb = self.model.forward(images)
            if self.metric in ('C', 'N'):
                emb = F.normalize(emb)
            # <-a, p->
            if self.metric in ('E', 'N'):
                loss = -F.pairwise_distance(emb[len(emb) // 2:],
                                            emb[:len(emb) // 2]).mean()
            elif self.metric in ('C',):
                loss = -1 + F.cosine_similarity(emb[len(emb) // 2:],
                                                emb[:len(emb) // 2]).mean()
            itermsg = {'loss': loss.item()}
            loss.backward()
            # projected gradient descent
            if self.pgditer > 1:
                images.grad.data.copy_(self.alpha * torch.sign(images.grad))
            elif self.pgditer == 1:
                images.grad.data.copy_(self.eps * torch.sign(images.grad))
            optx.step()
            images = torch.min(images, images_orig + self.eps)
            images = torch.max(images, images_orig - self.eps)
            images = torch.clamp(images, min=0., max=1.)
            images = images.clone().detach()
            images.requires_grad = True
            # report for the current iteration
            if self.verbose:
                print(f'(PGD)>', itermsg)
        # note: it is very important to clear the junk gradients.
        optm.zero_grad()
        optx.zero_grad()
        images.requires_grad = False
        # return the adversarial example
        if self.verbose:
            print(images.shape)
        return images

    def tribodycollapse(self, images: torch.Tensor, triplets: tuple):
        '''
        collapse (a, p, n) in the embedding space.
        '''
        # prepare
        anc, pos, neg = triplets
        imanc = images[anc, :, :, :].clone().detach().to(self.device)
        impos = images[pos, :, :, :].clone().detach().to(self.device)
        imneg = images[neg, :, :, :].clone().detach().to(self.device)
        images_orig = torch.cat([imanc, impos, imneg]).clone().detach()
        images = torch.cat([imanc, impos, imneg])
        images.requires_grad = True
        # start PGD
        self.model.eval()
        for iteration in range(self.pgditer):
            # optimizer
            optm = torch.optim.SGD(self.model.parameters(), lr=0.)
            optx = torch.optim.SGD([images], lr=1.)
            optm.zero_grad()
            optx.zero_grad()
            # forward data to be perturbed
            emb = self.model.forward(images)
            if self.metric in ('C', 'N'):
                emb = F.normalize(emb)
            ea = emb[:len(emb) // 3]
            ep = emb[len(emb) // 3:2 * len(emb) // 3]
            en = emb[2 * len(emb) // 3:]
            # draw two samples close to each other
            if self.metric in ('E', 'N'):
                loss = (F.pairwise_distance(ep, en) +
                        F.pairwise_distance(ea, ep) +
                        F.pairwise_distance(ea, en)).mean()
            elif self.metric in ('C',):
                loss = (3 - F.cosine_similarity(ep, en)
                          - F.cosine_similarity(ea, ep)
                          - F.cosine_similarity(ea, en)).mean()
            itermsg = {'loss': loss.item()}
            loss.backward()
            # projected gradient descent
            if self.pgditer > 1:
                images.grad.data.copy_(self.alpha * torch.sign(images.grad))
            elif self.pgditer == 1:
                images.grad.data.copy_(self.eps * torch.sign(images.grad))
            optx.step()
            images = torch.min(images, images_orig + self.eps)
            images = torch.max(images, images_orig - self.eps)
            images = torch.clamp(images, min=0., max=1.)
            images = images.clone().detach()
            images.requires_grad = True
            # report for the current iteration
            if self.verbose:
                print(f'(PGD)>', itermsg)
        # note: it is very important to clear the junk gradients.
        optm.zero_grad()
        optx.zero_grad()
        images.requires_grad = False
        # return the adversarial example
        if self.verbose:
            print(images.shape)
        return images

    def minmaxtriplet(self, images: torch.Tensor, triplets: tuple):
        '''
        Direct adaptation of Madry defense for triplet loss.
        Maximize triplet -> max dap, min dan. Modify all.
        '''
        # prepare
        anc, pos, neg = triplets
        imanc = images[anc, :, :, :].clone().detach().to(self.device)
        impos = images[pos, :, :, :].clone().detach().to(self.device)
        imneg = images[neg, :, :, :].clone().detach().to(self.device)
        images_orig = torch.cat([imanc, impos, imneg]).clone().detach()
        images = torch.cat([imanc, impos, imneg])
        images.requires_grad = True
        # start PGD
        self.model.eval()
        for iteration in range(self.pgditer):
            # optimizer
            optm = torch.optim.SGD(self.model.parameters(), lr=0.)
            optx = torch.optim.SGD([images], lr=1.)
            optm.zero_grad()
            optx.zero_grad()
            # forward data to be perturbed
            emb = self.model.forward(images)
            if self.metric in ('C', 'N'):
                emb = F.normalize(emb)
            ea = emb[:len(emb) // 3]
            ep = emb[len(emb) // 3:2 * len(emb) // 3]
            en = emb[2 * len(emb) // 3:]
            # draw two samples close to each other
            if self.metric in ('E', 'N'):
                loss = (F.pairwise_distance(ea, en) -
                        F.pairwise_distance(ea, ep)).mean()
            elif self.metric in ('C',):
                loss = ((1 - F.cosine_similarity(ea, en)) -
                        (1 - F.cosine_similarity(ea, ep))).mean()
            itermsg = {'loss': loss.item()}
            loss.backward()
            # projected gradient descent
            if self.pgditer > 1:
                images.grad.data.copy_(self.alpha * torch.sign(images.grad))
            elif self.pgditer == 1:
                images.grad.data.copy_(self.eps * torch.sign(images.grad))
            optx.step()
            images = torch.min(images, images_orig + self.eps)
            images = torch.max(images, images_orig - self.eps)
            images = torch.clamp(images, min=0., max=1.)
            images = images.clone().detach()
            images.requires_grad = True
            # report for the current iteration
            if self.verbose:
                print(f'(PGD)>', itermsg)
        # note: it is very important to clear the junk gradients.
        optm.zero_grad()
        optx.zero_grad()
        images.requires_grad = False
        # return the adversarial example
        if self.verbose:
            print(images.shape)
        return images

    def anticolanchperplex(self, images: torch.Tensor, triplets: tuple):
        '''
        collapse (p, n) and perplex (a).
        '''
        # prepare
        anc, pos, neg = triplets
        imanc = images[anc, :, :, :].clone().detach().to(self.device)
        impos = images[pos, :, :, :].clone().detach().to(self.device)
        imneg = images[neg, :, :, :].clone().detach().to(self.device)
        images_orig = torch.cat([imanc, impos, imneg]).clone().detach()
        images = torch.cat([imanc, impos, imneg])
        images.requires_grad = True
        # start PGD
        self.model.eval()
        for iteration in range(self.pgditer):
            # optimizer
            optm = torch.optim.SGD(self.model.parameters(), lr=0.)
            optx = torch.optim.SGD([images], lr=1.)
            optm.zero_grad()
            optx.zero_grad()
            # forward data to be perturbed
            emb = self.model.forward(images)
            if self.metric in ('C', 'N'):
                emb = F.normalize(emb)
            ea = emb[:len(emb) // 3]
            ep = emb[len(emb) // 3:2 * len(emb) // 3]
            en = emb[2 * len(emb) // 3:]
            # draw two samples close to each other and perplex the anchor
            if np.random.random() > 0.5:
                if self.metric in ('E', 'N'):
                    loss = F.pairwise_distance(ep, en).mean()
                elif self.metric in ('C',):
                    loss = (1 - F.cosine_similarity(ep, en)).mean()
            else:
                if self.metric in ('E', 'N'):
                    loss = (F.pairwise_distance(ea, en)
                            - F.pairwise_distance(ea, ep)).mean()
                elif self.metric in ('C',):
                    loss = ((1 - F.cosine_similarity(ea, en))
                            - (1 - F.cosine_similarity(ea, ep))).mean()
            itermsg = {'loss': loss.item()}
            loss.backward()
            # projected gradient descent
            if self.pgditer > 1:
                images.grad.data.copy_(self.alpha * torch.sign(images.grad))
            elif self.pgditer == 1:
                images.grad.data.copy_(self.eps * torch.sign(images.grad))
            optx.step()
            images = torch.min(images, images_orig + self.eps)
            images = torch.max(images, images_orig - self.eps)
            images = torch.clamp(images, min=0., max=1.)
            images = images.clone().detach()
            images.requires_grad = True
            # report for the current iteration
            if self.verbose:
                print(f'(PGD)>', itermsg)
        # note: it is very important to clear the junk gradients.
        optm.zero_grad()
        optx.zero_grad()
        images.requires_grad = False
        # return the adversarial example
        if self.verbose:
            print(images.shape)
        return images


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

