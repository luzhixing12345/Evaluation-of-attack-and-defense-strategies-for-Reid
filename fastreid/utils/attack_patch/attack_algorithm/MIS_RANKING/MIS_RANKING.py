
import numpy as np
import os.path as osp
from random import sample 
from scipy import io


import torch
import torch.nn as nn
import torch.optim as optim

from fastreid.engine.defaults import DefaultTrainer
from fastreid.modeling.heads.build import build_heads
from fastreid.utils.checkpoint import Checkpointer
from .GD import Generator, MS_Discriminator, Pat_Discriminator, GANLoss, ResnetG, weights_init
from .advloss import DeepSupervision, adv_CrossEntropyLoss, adv_CrossEntropyLabelSmooth, adv_TripletLoss

from fastreid.utils.reid_patch import change_preprocess_image, eval_ssim, get_train_set

is_training = False
Imagenet_mean = [0.485, 0.456, 0.406]
Imagenet_stddev = [0.229, 0.224, 0.225]
device = 'cuda'

def check_freezen(net, need_modified=False, after_modified=None):
  # print(net)
  cc = 0
  for child in net.children():
    for param in child.parameters():
      if need_modified: param.requires_grad = after_modified
      # if param.requires_grad: print('child', cc , 'was active')
      # else: print('child', cc , 'was forzen')
    cc += 1

def make_MIS_Ranking_generator(cfg,ak_type=-1,pretrained=False):
  # actually i couldn't understand why somebody write python code with two space indented instead of four???
  train_set = get_train_set(cfg)
  clf_criterion = adv_CrossEntropyLabelSmooth(num_classes=train_set.dataset.num_classes) if ak_type<0 else nn.MultiLabelSoftMarginLoss()
  metric_criterion = adv_TripletLoss(ak_type=ak_type)
  criterionGAN = GANLoss()   

  cfg = DefaultTrainer.auto_scale_hyperparams(cfg,train_set.dataset.num_classes)
  model = DefaultTrainer.build_model_main(cfg)  # use baseline_train
  model.preprocess_image=change_preprocess_image(cfg) # re-range the input size to [0,1]
  Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
  model.to(device)

  check_freezen(model, need_modified=True, after_modified=False)

  G = Generator(3, 3, 32, norm='bn').apply(weights_init)
  #G = ResnetG(3,3,32).apply(weights_init)
  D = MS_Discriminator(input_nc=6).apply(weights_init)
  #D = Pat_Discriminator(input_nc=6).apply(weights_init)
  check_freezen(G, need_modified=True, after_modified=True)
  check_freezen(D, need_modified=True, after_modified=True)
  optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
  optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

  model.to(device)
  G.to(device)
  D.to(device)
  G_save_pos = f'./model/G_weights_{cfg.DATASETS.NAMES[0]}_{cfg.CFGTYPE}.pth'
  D_save_pos = f'./model/D_weights_{cfg.DATASETS.NAMES[0]}_{cfg.CFGTYPE}.pth'

  
  EPOCH = 10
  if not pretrained:
    for epoch in range(EPOCH):
      train(cfg,epoch, G, D, model,criterionGAN, clf_criterion, metric_criterion, optimizer_G, optimizer_D, train_set,ak_type)
    
    torch.save(G.state_dict(), G_save_pos)
    torch.save(D.state_dict(), D_save_pos)
    print("successfully save attack model weights of G and D")
    
    MIS_Ranking_generator = generator(G,D,cfg)
  else:
    G.load_state_dict(torch.load(G_save_pos))
    D.load_state_dict(torch.load(D_save_pos))
    print("successfully load attack model weights of G and D")
    MIS_Ranking_generator = generator(G,D,cfg)

  return MIS_Ranking_generator


def train(cfg,epoch, G, D, model,criterionGAN, clf_criterion, metric_criterion, optimizer_G, optimizer_D, trainloader,ak_type):
  G.train()
  D.train()
  global is_training
  is_training = True

  loss_G_total = 0
  loss_D_total = 0
  print(f"start training epoch {epoch} for Mis-ranking model G and D")
  for batch_idx, data in enumerate(trainloader):
    if batch_idx>4000:
      break
    imgs = (data['images']/255).cuda()
    pids = data['targets'].cuda()

    imgs = imgs.clone().detach()
    imgs.requires_grad_()
    new_imgs, mask = perturb(imgs, G, D, cfg,train_or_test='train')
    if batch_idx%200==0:
      print(f'ssim = {eval_ssim(imgs,new_imgs)}')
    mask = mask.cuda()
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
    model.heads.mode = 'F'
    features = model(new_imgs)
    model.heads.mode = 'C'
    logits = model(new_imgs)

    new_outputs = logits
    new_features = features.view(features.size(0), -1)

    xent_loss, global_loss, loss_G_ssim = 0, 0, 0
    targets = None

    if ak_type < 0:
      xent_loss = DeepSupervision(clf_criterion, new_outputs, pids) if isinstance(new_features, (tuple, list)) else clf_criterion(new_outputs, pids)


    global_loss = DeepSupervision(metric_criterion, new_features, pids, targets) if isinstance(new_features, (tuple, list)) else metric_criterion(new_features, pids, targets)
    
    loss_G_ReID = (xent_loss+ global_loss)*10

    from .util.ms_ssim import msssim
    loss_func = msssim
    loss_G_ssim = (1-loss_func(imgs, new_imgs))*0.1


    ############## Forward ###############
    loss_D = (loss_D_fake + loss_D_real)/2
    loss_G = loss_G_GAN + loss_G_ReID + loss_G_ssim
    loss_G_total+=loss_G.item()
    loss_D_total+=loss_D.item()
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
  
  print(f'loss_G = {loss_G_total}')
  print(f'loss_D = {loss_D_total}')

def perturb(imgs, G, D, cfg,train_or_test='test'):
  n,c,h,w = imgs.size()
  delta = G(imgs)
  delta = L_norm(cfg,delta, train_or_test)
  new_imgs = torch.add(imgs.cuda(), delta[0:imgs.size(0)].cuda())

  _, mask = D(torch.cat((imgs, new_imgs.detach()), 1))
  delta = delta * mask
  new_imgs = torch.add(imgs.cuda(), delta[0:imgs.size(0)].cuda())

  for c in range(3):
    new_imgs.data[:,c,:,:] = new_imgs.data[:,c,:,:].clamp(new_imgs.data[:,c,:,:].min(), new_imgs.data[:,c,:,:].max()) # do clamping per channel
  if train_or_test == 'train':
    return new_imgs, mask
  elif train_or_test == 'test':
    return new_imgs, delta, mask

def L_norm(cfg,delta, mode='train'):

  delta.data += 1 
  delta.data *= 0.5

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


class generator:
  def __init__(self,G,D,cfg) -> None:
      
      self.G = G
      self.D = D
      self.cfg = cfg
      self.G.eval()
      self.D.eval()
  def __call__(self, images, y):
    # y is not used in SSAE, Just for universal adaptation
    if len(images.shape)==5:
        return self.GA(images)

    with torch.no_grad():
      new_imgs, _,_ = perturb(images, self.G, self.D,self.cfg, train_or_test='test')

    return new_imgs

  def GA(self,images):
    
    _,N,_,_,_ = images.shape

    new_images = []
    for i in range(N):
      img = images[:,i,:,:,:]
      with torch.no_grad():
        new_imgs, _,_ = perturb(img, self.G, self.D,self.cfg, train_or_test='test')
      new_images.append(new_imgs)
    
    new_images = torch.stack(new_images)
    new_images = new_images.permute(1,0,2,3,4)
    return new_images