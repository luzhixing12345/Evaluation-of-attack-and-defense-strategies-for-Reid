

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
import math
from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer
from fastreid.modeling.heads import build_heads
from fastreid.utils.reid_patch import get_train_set, save_image
import numpy as np

device  = 'cuda'
def check_freezen(net, need_modified=False, after_modified=None):
  for child in net.children():
    for param in child.parameters():
      if need_modified: param.requires_grad = after_modified
      # if param.requires_grad: print('child', cc , 'was active')
      # else: print('child', cc , 'was forzen')

def mis_ranking(cfg,queryloader,pos):
        ### Prepare criterion ###
    
    cfg = DefaultTrainer.auto_scale_hyperparams(cfg,queryloader.dataset.num_classes)
    model = DefaultTrainer.build_model_main(cfg)
    model.heads = build_heads(cfg)
    model.to(device)
    #Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

    
    clf_criterion = adv_CrossEntropyLabelSmooth(num_classes=queryloader.dataset.num_classes)
    metric_criterion = adv_TripletLoss(margin=0.3, ak_type=-1)
    criterionGAN = GANLoss()   

    ### Prepare pretrained model ###
    check_freezen(model, need_modified=True, after_modified=False)

    ### Prepare main net ###
    G = Generator(3, 3, 32, norm="bn").apply(weights_init)
    D = MS_Discriminator(input_nc=6).apply(weights_init)
    check_freezen(G, need_modified=True, after_modified=True)
    check_freezen(D, need_modified=True, after_modified=True)
    # setup optimizer

    optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    

    model = nn.DataParallel(model).cuda() 
    G = nn.DataParallel(G).cuda()
    D = nn.DataParallel(D).cuda()

    train(cfg,G, D, model, criterionGAN, clf_criterion, metric_criterion, optimizer_G, optimizer_D, queryloader,queryloader,pos)

    

def train(cfg,G, D, model, criterionGAN, clf_criterion, metric_criterion, optimizer_G, optimizer_D, trainloader,queryloader,pos):
  G.train()
  D.train()

  for idx,data in enumerate(trainloader):

    imgs = data['images'].to(device)
    pids = data['targets'].to(device)

    new_imgs= perturb(imgs, G, D,cfg)
    new_imgs = new_imgs.cuda()
    # Fake Detection and Loss
    pred_fake_pool, _ = D(torch.cat((imgs, new_imgs.detach()), 1))
    loss_D_fake = criterionGAN(pred_fake_pool, False)        

    # Real Detection and Loss
    num = cfg.SOLVER.IMS_PER_BATCH//2
    pred_real, _ = D(torch.cat((imgs[0:num,:,:,:], imgs[num:,:,:,:].detach()), 1))
    loss_D_real = criterionGAN(pred_real, True)

    # GAN loss (Fake Passability Loss)
    pred_fake, _ = D(torch.cat((imgs, new_imgs), 1))        
    loss_G_GAN = criterionGAN(pred_fake, True)               
    
    # Re-ID advloss
    model.train()
    #with torch.no_grad():
    outputs= model(new_imgs)
    new_outputs=outputs["pred_class_logits"]
    new_features=outputs['features']

    xent_loss, global_loss, loss_G_ssim = 0, 0, 0
    targets = None


    xent_loss = clf_criterion(new_outputs, pids)

    global_loss = metric_criterion(new_features, pids, targets)
    
    loss_G_ReID = (xent_loss+ global_loss) 


    ############## Forward ###############
    loss_D = (loss_D_fake + loss_D_real)/2
    loss_G = loss_G_GAN + loss_G_ReID + loss_G_ssim
    ############## Backward #############
    # update generator weights
    optimizer_G.zero_grad()
    # loss_G.backward(retain_graph=True)
    loss_G.backward()
    optimizer_G.step()
    # update discriminator weights
    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()

  extract_and_perturb(queryloader, G, D,cfg,pos)



def perturb(imgs, G, D, cfg):
  n,c,h,w = imgs.size()
  delta = G(imgs)
  delta = L_norm(delta,cfg)
  new_imgs = torch.add(imgs.cuda(), delta[0:imgs.size(0)].cuda())

  _, mask = D(torch.cat((imgs, new_imgs.detach()), 1))
  delta = delta * mask
  new_imgs = torch.add(imgs.cuda(), delta[0:imgs.size(0)].cuda())

  for c in range(3):
    new_imgs.data[:,c,:,:] = new_imgs.data[:,c,:,:].clamp(new_imgs.data[:,c,:,:].min(), new_imgs.data[:,c,:,:].max()) # do clamping per channel
  return new_imgs

def L_norm(delta,cfg):
  delta.data += 1 
  delta.data *= 0.5
  Imagenet_mean=cfg.MODEL.PIXEL_MEAN
  Imagenet_stddev=cfg.MODEL.PIXEL_STD
  for c in range(3):
    delta.data[:,c,:,:] = (delta.data[:,c,:,:] - Imagenet_mean[c]) / Imagenet_stddev[c]

  bs = cfg.SOLVER.IMS_PER_BATCH
  for i in range(bs):
    # do per channel l_inf normalization
    for ci in range(3):
      try:
        l_inf_channel = delta[i,ci,:,:].data.abs().max()
        # l_inf_channel = torch.norm(delta[i,ci,:,:]).data
        mag_in_scaled_c = 16/(255.0*Imagenet_stddev[ci])
        delta[i,ci,:,:].data *= np.minimum(1.0, mag_in_scaled_c / l_inf_channel.cpu()).float().cuda()
      except IndexError:
        break
  return delta


def extract_and_perturb(loader, G, D,cfg,pos):
  for _, data in enumerate(loader):
      imgs = data['images'].to(device)
      path = data['img_paths']
      G.eval()
      D.eval()
      new_imgs=perturb(imgs, G, D,cfg)
      save_image(new_imgs/255,path,pos)





import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.autograd import Variable
import math

class Pat_Discriminator(nn.Module):
  """
  Defines a PatchGAN discriminator
  Code based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
  """
  def __init__(self, input_nc, ndf=64, n_layers=3, norm='bn'):
    """Construct a PatchGAN discriminator
    Parameters:
        input_nc (int)  -- the number of channels in input images
        ndf (int)       -- the number of filters in the last conv layer
        n_layers (int)  -- the number of conv layers in the discriminator
        norm_layer      -- normalization layer
    """
    super(Pat_Discriminator, self).__init__()

    norm_layer = nn.BatchNorm2d if norm == 'bn' else nn.InstanceNorm2d
    if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
      use_bias = norm_layer.func != nn.BatchNorm2d
    else:
      use_bias = norm_layer != nn.BatchNorm2d

    kw = 4
    padw = 1
    sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
    nf_mult = 1
    nf_mult_prev = 1
    for n in range(1, n_layers):  # gradually increase the number of filters
      nf_mult_prev = nf_mult
      nf_mult = min(2 ** n, 8)
      sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]

    nf_mult_prev = nf_mult
    nf_mult = min(2 ** n_layers, 8)
    sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
    sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
    self.model = nn.Sequential(*sequence)

  def forward(self, x):
    return self.model(x), torch.ones_like(x)


class MS_Discriminator(nn.Module):
  def __init__(self, input_nc, ndf=64, n_layers=3, norm='bn', num_D=3, temperature=-1, use_gumbel=False):
    super(MS_Discriminator, self).__init__()
    self.num_D = num_D
    self.n_layers = n_layers
    self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, count_include_pad=False)
    self.same0 = SamePadding(kernel_size=3, stride=2)
    self.same1 = SamePadding(kernel_size=4, stride=2)
    self.same2 = SamePadding(kernel_size=4, stride=1)
    self.Mask = Mask(norm, temperature, use_gumbel)

    for i in range(num_D):
      netD = sub_Discriminator(input_nc, ndf, n_layers, norm)
      for j in range(n_layers+2): setattr(self, 'D'+str(i)+'_layer'+str(j), getattr(netD, 'layer'+str(j)))                                   

  def single_forward(self, model, x):
    result = [x]
    for i in range(len(model)): 
      samepadding = self.same1 if i < len(model)-2 else self.same2
      result.append(model[i](samepadding(result[-1])))
    return result[1:]

  def forward(self, x):        
    num_D = self.num_D
    proposal = []
    result = []
    mask = None
    input_downsampled = x
    for i in range(num_D):
      model = [getattr(self, 'D'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
      proposal.append(self.single_forward(model, input_downsampled)) #[[D2L0, D2L1,..., D2L4],[D1L0,...,D1L4],[D0L0,...,D0L4]]
      if i != (num_D-1): input_downsampled = self.downsample(self.same0(input_downsampled))
    for i in proposal: result.append(i[-1])
    mask = self.Mask(x, proposal)
    return result, mask
        
# (64,128,256,512,1) 
class sub_Discriminator(nn.Module):
  def __init__(self, input_nc, ndf=64, n_layers=3, norm='in'):
    super(sub_Discriminator, self).__init__()
    self.n_layers = n_layers

    use_bias = norm == 'in'
    norm_layer = nn.BatchNorm2d if norm == 'bn' else nn.InstanceNorm2d
    sequence = [[SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, bias=use_bias)), nn.LeakyReLU(0.2, True)]]
    nf = ndf
    for n in range(1, n_layers):
      nf_prev = nf
      nf = min(nf*2, 512)
      sequence += [[SpectralNorm(nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, bias=use_bias)), norm_layer(nf), nn.LeakyReLU(0.2, True)]]

    nf_prev = nf
    nf = min(nf*2, 512)
    sequence += [[SpectralNorm(nn.Conv2d(nf_prev, nf, kernel_size=4, stride=1, bias=use_bias)), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
    sequence += [[nn.Conv2d(nf, 1, kernel_size=4, stride=1)]]

    for n in range(len(sequence)):
      setattr(self, 'layer'+str(n), nn.Sequential(*sequence[n]))

  def forward(self, input):
    res = [input]
    for n in range(self.n_layers+2):
      model = getattr(self, 'layer'+str(n))
      res.append(model(res[-1]))
    return res[1:]

class Mask(nn.Module):
  def __init__(self, norm, temperature, use_gumbel, fused=1):
    super(Mask, self).__init__()
    self.temperature = temperature
    self.use_gumbel = use_gumbel
    self.fused = fused
    self.T = nn.Parameter(torch.Tensor([1]))
    norm_layer = nn.BatchNorm2d if norm == 'bn' else nn.InstanceNorm2d
    small_channels = [512, 512, 256, 128]
    big_channels = [512+128, 512+128+64, 128+64, 64] if self.fused == 2 else [512, 512, 128, 64]

    self.up32_16 = UpLayer(big_channels=big_channels[0], out_channels=512, small_channels=small_channels[0], norm_layer=norm_layer)
    self.up16_8 = UpLayer(big_channels=big_channels[1], out_channels=256, small_channels=small_channels[1], norm_layer=norm_layer)
    self.up8_4 = UpLayer(big_channels=big_channels[2], out_channels=128, small_channels=small_channels[2], norm_layer=norm_layer)
    # self.up4_2 = UpLayer(big_channels=big_channels[3], out_channels=64, small_channels=small_channels[3], norm_layer=norm_layer)
    self.deconv1 = nn.Sequential(*[SpectralNorm(nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)), nn.LeakyReLU(0.2, True)])
    self.deconv2 = nn.Sequential(*[SpectralNorm(nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)), nn.LeakyReLU(0.2, True)])
    self.conv2 = nn.Sequential(*[nn.Conv2d(128, 1, kernel_size=1, stride=1)])
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, x, proposal):
    n,c,h,w = x.size()
    if self.temperature == -1: return torch.ones((n,1,h,w)).cuda()
    scale32 = proposal[2][3]
    scale16 = torch.cat((proposal[2][1], proposal[1][3]),1) if self.fused == 2 else proposal[1][3]
    scale8 = torch.cat((proposal[0][3], proposal[1][1], proposal[2][0]),1) if self.fused == 2 else proposal[0][3]
    scale4 = torch.cat((proposal[0][1], proposal[1][0]),1) if self.fused == 2 else proposal[0][1]
    scale2 = proposal[0][0]
    out = self.up32_16(scale32, scale16)
    out = self.up16_8(out, scale8)
    out = self.up8_4(out, scale4)
    # out = self.up4_2(out, scale2)
    out = self.deconv1(out) 
    out = self.deconv2(out) 
    out = self.conv2(out)

    if not self.use_gumbel:
      logits = self.logsoftmax(out.view(n, -1))
      th, _ = torch.topk(logits, k=int(self.temperature), dim=1, largest=True)
      mask, zeros, ones = torch.zeros_like(logits).cuda(), torch.zeros(h*w).cuda(), torch.ones(h*w).cuda()
      for i in range(n):
        mask[i,:] = torch.where(logits[i,:]>=th[i, int(self.temperature)-1], ones, zeros)
      mask = mask.view(n, 1, h, w)
    elif self.use_gumbel:
      logits = gumbel_softmax(out.view(n, -1), k=int(self.temperature), T=self.T, hard=True, eps=1e-10).view(n, 1, h, w)
      mask = logits.cuda()
      # logits = F.gumbel_softmax(out.view(n, -1), tau=self.temperature).view(n, 1, h, w)
      # # logits_normed = torch.clamp((logits_normed+1e-4), min=0, max=1)
      # logits = np.minimum(1.0, logits.data.cpu().numpy()*(h*w)+1e-4)
      # mask = torch.bernoulli(torch.from_numpy(logits)).cuda()
    return mask
    

def _sample_gumbel(shape, eps=1e-10, out=None):
  """
  Based on
  https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
  (MIT license)
  """
  U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
  return - torch.log(eps - torch.log(U + eps))

def _gumbel_softmax_sample(logits, T=1, eps=1e-10):
  """
  Based on
  https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
  (MIT license)
  """
  dims = logits.dim()
  gumbel_noise = _sample_gumbel(logits.size(), eps=eps, out=logits.data.new())
  y = logits + gumbel_noise
  return F.softmax(y / T, dims - 1)
def gumbel_softmax(logits, k, T=1, hard=True, eps=1e-10):
  shape = logits.size()
  assert len(shape) == 2
  y_soft = _gumbel_softmax_sample(logits, T=T, eps=eps)
  if hard:
    _, ind = torch.topk(y_soft, k=k, dim=-1, largest=True)
    y_hard = logits.new_zeros(*shape).scatter_(-1, ind.view(-1, k), 1.0)
    y = y_hard - y_soft.detach() + y_soft
  else:
    y = y_soft
  return y
  
class UpLayer(nn.Module):
  def __init__(self, big_channels, out_channels, small_channels, norm_layer):
    super(UpLayer, self).__init__()
    self.big_channels = big_channels
    self.out_channels = out_channels
    self.small_channels = small_channels
    self.conv1 = nn.Sequential(*[SpectralNorm(nn.Conv2d(self.big_channels, self.small_channels, kernel_size=1, stride=1)), norm_layer(self.small_channels), nn.LeakyReLU(0.2, True)])
    self.conv2 = nn.Sequential(*[SpectralNorm(nn.Conv2d(self.small_channels, self.out_channels, kernel_size=3, stride=1, padding=1)), norm_layer(self.out_channels), nn.LeakyReLU(0.2, True)])

  def forward(self, small, big):
    small = F.upsample(small, size=(big.size()[2], big.size()[3]), mode='bilinear')
    big = self.conv1(big)
    out = self.conv2(big+small)
    return out

class Generator(nn.Module):
  def __init__(self, input_nc, output_nc, ngf, norm='bn', n_blocks=6):
    super(Generator, self).__init__()

    n_downsampling = n_upsampling = 2
    use_bias = norm == 'in'
    norm_layer = nn.BatchNorm2d if norm == 'bn' else nn.InstanceNorm2d
    begin_layers, down_layers, res_layers, up_layers, end_layers = [], [], [], [], []
    for i in range(n_upsampling): 
      up_layers.append([])
    # ngf
    begin_layers = [nn.ReflectionPad2d(3), SpectralNorm(nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)), norm_layer(ngf), nn.ReLU(True)]
    # 2ngf, 4ngf
    for i in range(n_downsampling):
      mult = 2**i
      down_layers += [SpectralNorm(nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1, bias=use_bias)), norm_layer(ngf*mult*2), nn.ReLU(True)]
    # 4ngf
    mult = 2**n_downsampling
    for i in range(n_blocks):
      res_layers += [ResnetBlock(ngf*mult, norm_layer, use_bias)]
    # 2ngf, ngf
    for i in range(n_upsampling):
      mult = 2**(n_upsampling - i)
      up_layers[i] += [SpectralNorm(nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)), norm_layer(int(ngf*mult/2)), nn.ReLU(True)]

    end_layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

    self.l1 = nn.Sequential(*begin_layers)
    self.l2 = nn.Sequential(*down_layers)
    self.l3 = nn.Sequential(*res_layers)
    self.l4_1 = nn.Sequential(*up_layers[0])
    self.l4_2 = nn.Sequential(*up_layers[1])
    self.l5 = nn.Sequential(*end_layers)

  def forward(self, inputs):
    out = self.l1(inputs)
    out = self.l2(out)
    out = self.l3(out)
    out = self.l4_1(out)
    out = self.l4_2(out)
    out = self.l5(out)
    return out

class ResnetG(nn.Module):
  def __init__(self, input_nc, output_nc, ngf, norm='bn', n_blocks=6):
    super(ResnetG, self).__init__()

    n_downsampling = n_upsampling = 2
    use_bias = norm == 'in'
    norm_layer = nn.BatchNorm2d if norm == 'bn' else nn.InstanceNorm2d
    begin_layers, down_layers, res_layers, up_layers, end_layers = [], [], [], [], []
    for i in range(n_upsampling): 
      up_layers.append([])
    # ngf
    begin_layers = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
    # 2ngf, 4ngf
    for i in range(n_downsampling):
      mult = 2**i
      down_layers += [nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf*mult*2), nn.ReLU(True)]
    # 4ngf
    mult = 2**n_downsampling
    for i in range(n_blocks):
      res_layers += [ResnetBlock(ngf*mult, norm_layer, use_bias)]
    # 2ngf, ngf
    for i in range(n_upsampling):
      mult = 2**(n_upsampling - i)
      up_layers[i] += [nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias), norm_layer(int(ngf*mult/2)), nn.ReLU(True)]

    end_layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

    self.l1 = nn.Sequential(*begin_layers)
    self.l2 = nn.Sequential(*down_layers)
    self.l3 = nn.Sequential(*res_layers)
    self.l4_1 = nn.Sequential(*up_layers[0])
    self.l4_2 = nn.Sequential(*up_layers[1])
    self.l5 = nn.Sequential(*end_layers)

  def forward(self, inputs):
    out = self.l1(inputs)
    out = self.l2(out)
    out = self.l3(out)
    out = self.l4_1(out)
    out = self.l4_2(out)
    out = self.l5(out)
    return out

# Define a resnet block
class ResnetBlock(nn.Module):
  def __init__(self, dim, norm_layer, use_bias):
    super(ResnetBlock, self).__init__()
    self.conv_block = self.build_conv_block(dim, norm_layer, use_bias)

  def build_conv_block(self, dim, norm_layer, use_bias):
    conv_block = []
    for i in range(2):
      conv_block += [nn.ReflectionPad2d(1)]
      conv_block += [SpectralNorm(nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias)), norm_layer(dim)]
      if i < 1: 
        conv_block += [nn.ReLU(True)]
    return nn.Sequential(*conv_block)

  def forward(self, x):
    out = x + self.conv_block(x)
    return out

class SamePadding(nn.Module):
  def __init__(self, kernel_size, stride):
    super(SamePadding, self).__init__()
    self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
    self.stride = torch.nn.modules.utils._pair(stride)

  def forward(self, input):
    in_width = input.size()[2]
    in_height = input.size()[3]
    out_width = math.ceil(float(in_width) / float(self.stride[0]))
    out_height = math.ceil(float(in_height) / float(self.stride[1]))
    pad_along_width = ((out_width - 1) * self.stride[0] +
                       self.kernel_size[0] - in_width)
    pad_along_height = ((out_height - 1) * self.stride[1] +
                        self.kernel_size[1] - in_height)
    pad_left = int(pad_along_width / 2)
    pad_top = int(pad_along_height / 2)
    pad_right = pad_along_width - pad_left
    pad_bottom = pad_along_height - pad_top
    return F.pad(input, (int(pad_left), int(pad_right), int(pad_top), int(pad_bottom)), 'constant', 0)

  def __repr__(self):
    return self.__class__.__name__

def weights_init(m):
  classname = m.__class__.__name__
  # print(dir(m))
  if classname.find('Conv') != -1:
    if 'weight' in dir(m): 
      m.weight.data.normal_(0.0, 1)
  elif classname.find('BatchNorm2d') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)

class GANLoss(nn.Module):
  def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.cuda.FloatTensor):
    super(GANLoss, self).__init__()
    self.real_label = target_real_label
    self.fake_label = target_fake_label
    self.real_label_var = None
    self.fake_label_var = None
    self.Tensor = tensor
    if use_lsgan: self.loss = nn.MSELoss()
    else: self.loss = nn.BCELoss()

  def get_target_tensor(self, input, target_is_real):
    target_tensor = None
    if target_is_real:
      create_label = ((self.real_label_var is None) or
                      (self.real_label_var.numel() != input.numel()))
      if create_label:
        real_tensor = self.Tensor(input.size()).fill_(self.real_label)
        self.real_label_var = Variable(real_tensor, requires_grad=False)
      target_tensor = self.real_label_var
    else:
      create_label = ((self.fake_label_var is None) or
                      (self.fake_label_var.numel() != input.numel()))
      if create_label:
        fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
        self.fake_label_var = Variable(fake_tensor, requires_grad=False)
      target_tensor = self.fake_label_var
    return target_tensor

  def __call__(self, input, target_is_real):
    if isinstance(input[0], list):
      loss = 0
      for input_i in input:
        pred = input_i[-1]
        target_tensor = self.get_target_tensor(pred, target_is_real)
        loss += self.loss(pred, target_tensor)
      return loss
    else:            
      target_tensor = self.get_target_tensor(input[-1], target_is_real)
      return self.loss(input[-1], target_tensor)

class adv_CrossEntropyLabelSmooth(nn.Module):
  """
  Args:
      num_classes (int): number of classes.
      epsilon (float): weight.
  """
  def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
    super(adv_CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.use_gpu = use_gpu
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, logits, pids):
    """
    Args:
        logits: prediction matrix (before softmax) with shape (batch_size, num_classes)
        pids: ground truth labels with shape (num_classes)
    """
    # n = pids.size(0)
    # _, top2 = torch.topk(logits, k=2, dim=1, largest=False)
    # adv_target = top2[:,0]
    # for i in range(n):
    #   if adv_target[i] == pids[i]: adv_target[i] = top2[i,1]
    #   else: continue
    _, adv_target = torch.min(logits, 1)
    # for i in range(n):
    #   while adv_target[i] == pids[i]:
    #     adv_target[i] = random.randint(0, self.num_classes)

    log_probs = self.logsoftmax(logits)
    adv_target = torch.zeros(log_probs.size()).scatter_(1, adv_target.unsqueeze(1).data.cpu(), 1)
    smooth = torch.ones(log_probs.size()) / (self.num_classes-1)
    smooth[:, pids.data.cpu()] = 0 # Pytorch1.0
    smooth = smooth.cuda()
    if self.use_gpu: adv_target = adv_target.cuda()
    adv_target = (1 - self.epsilon) * adv_target + self.epsilon * smooth
    loss = (- adv_target * log_probs).mean(0).sum()
    return torch.log(loss)


class adv_TripletLoss(nn.Module):
  def __init__(self, ak_type, margin=0.3):
    super(adv_TripletLoss, self).__init__()
    self.margin = margin
    self.ak_type = ak_type
    self.ranking_loss = nn.MarginRankingLoss(margin=margin)

  def forward(self, features, pids, targets=None):
      """
      Args:
          features: feature matrix with shape (batch_size, feat_dim)
          pids: ground truth labels with shape (num_classes)
          targets: pids with certain attribute (batch_size, pids)
      """
      n = features.size(0)

      dist = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
      dist = dist + dist.t()
      dist.addmm_(1, -2, features, features.t())
      dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

      if self.ak_type < 0: 
        mask = pids.expand(n, n).eq(pids.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
          dist_an.append(dist[i][mask[i]].min().unsqueeze(0)) # make nearest pos-pos far away
          dist_ap.append(dist[i][mask[i] == 0].max().unsqueeze(0)) # make hardest pos-neg closer

      elif self.ak_type > 0: 
        p = []
        for i in range(n):
          p.append(pids[i].item())
        mask = targets[0][p].expand(n, n).eq(targets[0][p].expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
          dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
          dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))

      dist_ap = torch.cat(dist_ap)
      dist_an = torch.cat(dist_an)

      y = torch.ones_like(dist_an)

      loss = self.ranking_loss(dist_an, dist_ap, y)
      return torch.log(loss)

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)