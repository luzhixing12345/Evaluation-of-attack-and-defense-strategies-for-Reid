
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.autograd import Variable

attack_img = Variable(torch.rand(3, 256, 128), requires_grad=True)*1e-6

def attack_update(att_img, grad, pre_sat, g, rate=0.8, base=False, i=10, radiu=10):

    norm = torch.sum(torch.abs(grad).view((grad.shape[0], -1)), dim=1).view(-1, 1, 1) + torch.tensor([[[1e-12]], [[1e-12]], [[1e-12]]])
    # norm = torch.max(torch.abs(grad).flatten())
    x_grad = grad / norm
    if torch.isnan(x_grad).any() or torch.isnan(g).any():
        import pdb
        pdb.set_trace()
    g = 0.4*g + x_grad
    att_img = att_img - 0.004*g.sign()
    radiu = radiu / 255.
    att_img = torch.clamp(att_img, -radiu, radiu)

    pre_sat = torch.div(torch.sum(torch.eq(torch.abs(att_img), radiu), dtype=torch.float32),
                    torch.tensor(att_img.flatten().size(), dtype=torch.float32))

    if not base:
        img_abs = torch.abs(att_img)
        img_sort = torch.sort(img_abs.flatten(), descending=True)[0]
        new_rate = max(pre_sat, rate)
        if pre_sat < rate and i > 0:
            img_median = img_sort[int((len(img_sort)*new_rate))]
            att_img = att_img * (radiu / (img_median + 1e-6))
            att_img = torch.clamp(att_img, -radiu, radiu)

    sat = torch.div(torch.sum(torch.eq(torch.abs(att_img), radiu), dtype=torch.float32),
                torch.tensor(att_img.flatten().size(), dtype=torch.float32))


    return att_img, sat, g





class MUAP:
    def __init__(self,cfg,model) -> None:
        self.cfg = cfg
        self.model = model
        self.model.eval()
        self.scale_rate = 0.8
        self.radiu = 10
        self.EPOCH = 50

        torch.manual_seed(1)
        
        self.normalize_transform = T.Normalize(mean=cfg.MODEL.PIXEL_MEAN, std=cfg.MODEL.PIXEL_STD)
        self.g = torch.tensor([0.])
        self.pre_sat = 1.
        self.loss_fn1 = MapLoss()
        self.loss_fn2 = TVLoss(TVLoss_weight=10.)

    def __call__(self, images,target):
        if len(images.shape)==5:
            return self.GA(images,target)
        
        global attack_img
        for epoch in range(self.EPOCH):
            attack_img = Variable(attack_img, requires_grad=True)
            median_img = torch.add(images,self.normalize_transform(attack_img).to('cuda')).to('cuda')   #mix attack img and clean img
            feat = self.model(images)
            attack_feat = self.model(median_img)
           
            map_loss = self.loss_fn1(attack_feat, feat, target, 5)
            tvl_loss = self.loss_fn2(median_img)
            total_loss = tvl_loss + map_loss

            total_loss.backward()
            self.model.zero_grad()

            attack_grad = attack_img.grad.data
            attack_img, sat, g = attack_update(attack_img, attack_grad, pre_sat, g, self.scale_rate,epoch,self.radiu)
            pre_sat = sat

        return attack_img
    
    def GA(self,images,target):

        global attack_img

        _,N,_,_,_ = images.shape

        new_images = []
        for i in range(N):
            img = images[:,i,:,:,:]
            for epoch in range(self.EPOCH):
                attack_img = Variable(attack_img, requires_grad=True)
                median_img = torch.add(img,self.normalize_transform(attack_img).to('cuda')).to('cuda')   #mix attack img and clean img
                feat = self.model(img)
                attack_feat = self.model(median_img)
            
                map_loss = self.loss_fn1(attack_feat, feat, target, 5)
                tvl_loss = self.loss_fn2(median_img)
                total_loss = tvl_loss + map_loss

                total_loss.backward()
                self.model.zero_grad()

                attack_grad = attack_img.grad.data
                attack_img, sat, g = attack_update(attack_img, attack_grad, pre_sat, g, self.scale_rate,epoch,self.radiu)
                pre_sat = sat
            new_images.append(img)
        
        new_images = torch.stack(new_images)
        new_images = new_images.permute(1,0,2,3,4)
        return new_images




def euclidean_dist(x, y, square=False):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    if square:
        dist = dist.clamp(min=1e-12)
    else:
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist



class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :]-x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:]-x[:, :, :, :w_x-1]), 2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size


class TVLossTmp(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLossTmp,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[1]
        w_x = x.size()[2]
        count_h = (x.size()[1]-1) * x.size()[2]
        count_w = x.size()[1] * (x.size()[2] - 1)
        h_tv = torch.pow((x[:, 1:, :]-x[:, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, 1:]-x[:, :, :w_x-1]), 2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

def normalize(x, axis=1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def map_loss_v3(atta_feat, feat, target, bin_num, margin=0):    # https://arxiv.org/pdf/1906.07589.pdf
    # import numpy as np
    # np.save('atta_feat.npy', atta_feat.data.cpu().numpy())
    # np.save('feat.npy', feat.data.cpu().numpy())
    # np.save('target.npy', target.data.cpu().numpy())
    N = atta_feat.size(0)
    atta_feat = normalize(atta_feat)
    feat = normalize(feat)
    dist_raw = euclidean_dist(atta_feat, feat)
    dist = dist_raw.clone()
    # bin_num = 20
    # bin_len = 2./(bin_num-1)
    bin_len = (2.+margin) / (bin_num - 1)
    is_pos = target.expand(N, N).eq(target.expand(N, N).t()).float()
    is_neg = target.expand(N, N).ne(target.expand(N, N).t()).float()
    total_true_indicator = torch.zeros(N).to('cuda')
    total_all_indicator = torch.zeros(N).to('cuda')
    AP = torch.zeros(N).to('cuda')

    # import pdb
    # pdb.set_trace()
    if margin is None:
        pass
    else:
        is_pos_index = target.expand(N, N).eq(target.expand(N, N).t())
        is_neg_index = target.expand(N, N).ne(target.expand(N, N).t())
        dist[is_pos_index] = dist_raw[is_pos_index] - margin/2
        dist[is_neg_index] = dist_raw[is_neg_index] + margin/2

    for i in range(1, bin_num+1):
        # bm = 1 - (i-1) * bin_len
        bm = (i-1) * bin_len - margin /2.
        indicator = (1 - torch.abs(dist - bm)/bin_len).clamp(min=0)
        true_indicator = is_pos * indicator
        all_indicator = indicator
        sum_true_indicator = torch.sum(true_indicator, 1)
        sum_all_indicator = torch.sum(all_indicator, 1)
        total_true_indicator = total_true_indicator + sum_true_indicator
        total_all_indicator = total_all_indicator + sum_all_indicator
        Pm = total_true_indicator / total_all_indicator.clamp(min=1e-12)
        rm = sum_true_indicator / 4
        ap_bin = Pm*rm
        AP = AP + ap_bin
        # import pdb
        # pdb.set_trace()
    final_AP = torch.sum(AP) / N
    return final_AP

class MapLoss(nn.Module):

    def __init__(self):
        super(MapLoss, self).__init__()
        # self.name = 'map'

    def forward(self, atta_feat, feat, target, bin_num, margin=0):
        loss = map_loss_v3(atta_feat, feat, target, bin_num, margin=margin)
        return loss


class ODFALoss(nn.Module):

    def __init__(self, use_gpu=True):
        super(ODFALoss, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, features_adv, features_x):
        '''
        Args:
            features_x: feature matrix with shape (feat_dim, ).
        '''
        assert features_adv.shape == features_x.shape
        # batch_size = features_x
        features_adv = features_adv.view(-1, 1)
        features_x = features_x.view(-1, 1)
        loss = torch.mm((features_adv / torch.norm(features_adv) + features_x / torch.norm(features_x)).t(),
                        (features_adv / torch.norm(features_adv) + features_x / torch.norm(features_x)))
        return loss