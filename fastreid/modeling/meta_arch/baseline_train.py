# encoding: utf-8
"""
@author: zhangshengyao
"""

import torch
from torch import nn

from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_train_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class Baseline_train(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

        # backbone
        self.backbone = build_backbone(cfg)

        # head
        self.heads = build_train_heads(cfg)

    @property
    def device(self):
        return self.pixel_mean.device


    '''
    the purpose of this baseline_train:
        1. under the train mode of baseline, it will directly return the loss, but actually 
           we need the logits value for classifier training 
        2. under the test mode of baseline, it will return the features of image, and at that time
            we need to change the self.heads to embedding_head instead of training_head
        3. in the code of advertorch.attack and advertorch.defense, although all methods give the
            arguments clip_max and clip_min, it actually doesn't work well if the data input
            in out of range 0.0~1.0, so we have to rerange the image data to adapt the input range.
            As a result, you can see sometimes the input is "data/255.0".
            But don't worry about that because it will be multiplied in preprocess_image funtion below
            so the actual input to self.backbone doesn't change
    
    '''

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)
        if isinstance(batched_inputs, torch.Tensor): 
            targets = None
        else :
            targets = batched_inputs["targets"]
        outputs = self.heads(features, targets)     
        return outputs
    
    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))
        
        images.sub(self.pixel_mean).div(self.pixel_std)
        return images