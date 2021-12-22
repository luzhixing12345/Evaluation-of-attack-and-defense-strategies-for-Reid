# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import Parameter
from copy import deepcopy
import torch.optim as optim

from fastreid.utils.reid_patch import get_train_set
device = 'cuda'
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

class SSAE(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=32, norm='bn', n_blocks=6):
        super(SSAE, self).__init__()

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
        end_layers2 = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, kernel_size=7, padding=0), nn.Sigmoid()]

        self.encoder_1 = nn.Sequential(*begin_layers)
        self.encoder_2 = nn.Sequential(*down_layers)
        self.encoder_3 = nn.Sequential(*res_layers)

        self.perturb_decoder_1 = nn.Sequential(*up_layers[0])
        self.perturb_decoder_2 = nn.Sequential(*up_layers[1])
        self.perturb_decoder_3 = nn.Sequential(*end_layers)

        self.mask_decoder_1 = deepcopy(self.perturb_decoder_1)
        self.mask_decoder_2 = deepcopy(self.perturb_decoder_2)
        self.mask_decoder_3 = nn.Sequential(*end_layers2)
        # self.set_requires_grad(self)  # this line cause bug

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self, x):
        # encoding
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        latency = self.encoder_3(x)

        # decoding
        delta = self.perturb_decoder_1(latency)
        delta = self.perturb_decoder_2(delta)
        delta = self.perturb_decoder_3(delta)

        mask = self.mask_decoder_1(latency)
        mask = self.mask_decoder_2(mask)
        mask = self.mask_decoder_3(mask)

        return delta, mask

def _batch_clamp_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following
    for ii in range(len(vector)):
        batch_tensor[ii] = clamp(
            batch_tensor[ii], -vector[ii], vector[ii])
    """
    return torch.min(
        torch.max(batch_tensor.transpose(0, -1), -vector), vector
    ).transpose(0, -1).contiguous()

def clamp(input, min=None, max=None):
    if min is not None and max is not None:
        return torch.clamp(input, min=min, max=max)
    elif min is None and max is None:
        return input
    elif min is None and max is not None:
        return torch.clamp(input, max=max)
    elif min is not None and max is None:
        return torch.clamp(input, min=min)
    else:
        raise ValueError("This is impossible")
def batch_clamp(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_clamp_tensor_by_vector(float_or_vector, tensor)
        return tensor
    elif isinstance(float_or_vector, float):
        tensor = clamp(tensor, -float_or_vector, float_or_vector)
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor
class CosineLoss(nn.Module):
    def __init__(self, ):
        super(CosineLoss, self).__init__()
    

    def forward(self, x, y):
        return ((torch.cosine_similarity(x, y, dim=1)+1)).sum()
class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Generator:
    def __init__(self,generator) -> None:
        self.generator = generator
        self.generator.eval()
        self.delta = 0.1
        self.clip_min = 0.0
        self.clip_max = 1.0

    def __call__(self, images, y) :
        # y is not used in SSAE, Just for universal adaptation
        if len(images.shape)==5:
            return self.GA(images)

        with torch.no_grad():
            perturbations, saliency_map = self.generator(images)

        images = images.to(device)
        perturbations = perturbations.detach().to(device)
        saliency_map = saliency_map.detach().to(device)
        
        adv_imgs = images + batch_clamp(self.delta, perturbations) * saliency_map
        adv_imgs = torch.clamp(adv_imgs, min=self.clip_min, max=self.clip_max)

        print('adv-ori',adv_imgs-images)
        return adv_imgs

    def GA(self,images):

        _,N,_,_,_ = images.shape

        new_images = []
        for i in range(N):
            img = images[:,i,:,:,:].clone()
            with torch.no_grad():
                perturbations, saliency_map = self.generator(img)

            img = img.to(device)
            perturbations = perturbations.detach().to(device)
            saliency_map = saliency_map.detach().to(device)
        
            adv_imgs = img + batch_clamp(self.delta, perturbations) * saliency_map
            adv_imgs = torch.clamp(adv_imgs, min=self.clip_min, max=self.clip_max)
            new_images.append(adv_imgs)

        new_images = torch.stack(new_images)
        new_images = new_images.permute(1,0,2,3,4)
        return new_images
        
def make_SSAE_generator(cfg,model,pretrained=False):

    generator = SSAE().to(device)
    generator = nn.DataParallel(generator)
    save_pos = './model/SSAE_generator_weights.pth'
    generator.train()
    train_loader = get_train_set(cfg)
    cosine_loss = CosineLoss().to(device)
    optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss(reduction='sum').to(device)

    EPOCHS = 20
    delta = 0.1
    alpha = 0.0001
    model.eval()
    if not pretrained:
        for epoch in range(EPOCHS):

            print(f'start epoch {epoch} training for SSAE')
            generator.train()

            for batch_idx, data in enumerate(train_loader):
            
                if batch_idx>4000:
                    break
                
                raw_imgs= (data['images']/255.0).to(device)
                # perturb images
                
                perturbations, saliency_map = generator(raw_imgs)
                perturbations = batch_clamp(delta, perturbations)

                adv_imgs =  raw_imgs + perturbations
                # extract features from imgs and adv_imgs
                raw_feats = model(raw_imgs)
                raw_norms = torch.norm(raw_feats, dim=1)
                raw_feats = nn.functional.normalize(raw_feats, dim=1, p=2)
                adv_feats = model(adv_imgs)
                adv_norms = torch.norm(adv_feats, dim=1)
                adv_feats = nn.functional.normalize(adv_feats, dim=1, p=2)

                angular_loss = cosine_loss(raw_feats, adv_feats)
                loss = angular_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        optimizer = optim.Adam(generator.parameters(), lr=1e-4)
        for epoch in range(EPOCHS):

            print(f'start epoch {epoch} training for SSAE')
            generator.train()

            for batch_idx, data in enumerate(train_loader):
            
                if batch_idx>4000:
                    break
                
                raw_imgs= (data['images']/255.0).to(device)
                # perturb images
                
                perturbations, saliency_map = generator(raw_imgs)
                perturbations = batch_clamp(delta, perturbations)

                adv_imgs =  raw_imgs + perturbations * saliency_map

                # extract features from imgs and adv_imgs
                raw_feats = model(raw_imgs)
                raw_norms = torch.norm(raw_feats, dim=1)
                raw_feats = nn.functional.normalize(raw_feats, dim=1, p=2)
                adv_feats = model(adv_imgs)
                adv_norms = torch.norm(adv_feats, dim=1)
                adv_feats = nn.functional.normalize(adv_feats, dim=1, p=2)

                angular_loss = cosine_loss(raw_feats, adv_feats)
                loss = angular_loss

                norm_loss = mse_loss(raw_norms, adv_norms)
                frobenius_loss = torch.norm(saliency_map, dim=(1,2), p=2).sum()
                loss += alpha * (norm_loss + frobenius_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        torch.save(generator.state_dict(), save_pos)
        print(f'save model weights in {save_pos}')
    else:
        generator.load_state_dict(torch.load(save_pos))
        print(f'load model weights from {save_pos}')
        
    SSAE_generator = Generator(generator)
    return SSAE_generator