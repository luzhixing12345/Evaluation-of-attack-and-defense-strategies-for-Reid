
import torch

from fastreid.utils.reid_patch import save_image
import statistics

device='cuda'
def CA(cfg,query_data_loader,model,pos):
    '''
    Note, all images must lie in [0,1]^D
    '''
    # prepare torche current batch of data
    candi = recompute_valvecs(model,query_data_loader)
    # XXX: this is tricky, but we need it.

    # initialize attacker
    advrank = AdvRank(model)

    for _,data in enumerate(query_data_loader):
        images = (data['images']/255).to(device)
        labels = data['targets'].to(device)
        path = data['img_paths']
        adv_data = advrank(images, labels, candi)
        save_image(adv_data,path,pos)


class AdvRank:
    def __init__(self,model) -> None:
        self.model=model
    def __call__(self, images: torch.Tensor, labels: torch.Tensor,
               candi: tuple) -> tuple:
        '''
        Note, all images must lie in [0,1]^D
        '''
        # prepare the current batch of data
        images = images.clone().detach()
        images.requires_grad = True

        self.model.eval()
        for iteration in range(self.pgditer):
            # >> prepare optimizer for SGD
            optim = torch.optim.SGD(self.model.parameters(), lr=1.)
            optimx = torch.optim.SGD([images], lr=1.)
            optim.zero_grad()
            optimx.zero_grad()
            output = self.model(images)
            embpairs, _ = self.qcsel
            loss, _ = AdvRankLoss(output, embpairs, candi[0])

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

def recompute_valvecs(model,query_data_loader):
    '''
    Compute embedding vectors for the whole validation dataset,
    in order to do image retrieval and evaluate the Recall@K scores,
    etc.
    '''
    with torch.no_grad():
        valvecs, vallabs = [], []
        for _, data in enumerate(query_data_loader):
            images = (data['images']/255).to(device)
            labels = data['targets'].to(device)
            features = model(images)
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
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
    return (valvecs, vallabs)