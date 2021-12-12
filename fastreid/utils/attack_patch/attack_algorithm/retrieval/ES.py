import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ES:
    def __init__(self,cfg,model) -> None:
        self.cfg = cfg
        self.model = model
        self.alpha = 0.01
        self.pgditer =24

    def __call__(self, images):
        if len(images.shape)==5:
            return self.GA(images)

        with torch.no_grad():
            features_origin = self.model(images)

        images = images.clone().detach()
        images.requires_grad_()
        for iteration in range(self.pgditer):
            # >> prepare optimizer for SGD
            optim = torch.optim.SGD(self.model.parameters(), lr=1.)
            optimx = torch.optim.SGD([images], lr=1.)
            optim.zero_grad()
            optimx.zero_grad()
            features = self.model(images)
            distance = F.pairwise_distance(features, features_origin)
            loss = -distance.sum()
            loss.backward()

            images.grad.data.copy_(self.alpha * torch.sign(images.grad))
 
            optimx.step()

        optim.zero_grad()
        optimx.zero_grad()
        return images


    def GA(self,images):
        pass