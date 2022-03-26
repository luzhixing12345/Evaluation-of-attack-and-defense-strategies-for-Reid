
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from fastreid.utils.attack_patch.attack_algorithm import IFGSM

class FNA:
    def __init__(self,cfg,model,direction) -> None:
        self.cfg = cfg
        self.model = model
        self.eps=0.05
        self.eps_iter=1.0/255.0
        self.target = direction
        self.AttackMethod = IFGSM (self.cfg,self.model,self.loss_fun, eps=self.eps, eps_iter=self.eps_iter,targeted=self.target,rand_init=False)

    def loss_fun(self,f1s,f2s):
        mse = nn.MSELoss(reduction='sum')
        m = 0
        for f1, f2 in zip(f1s, f2s):
            for i in range(len(f2)-1):
                m += mse(f1,f2[i])
            m -= mse(f1,f2[-1])
        return m

    def __call__(self, images, selected_features):
        if len(images)==5:
            return self.GA(images,selected_features)
        return self.AttackMethod(images,selected_features)

    def GA(self,selected_images,features):
        
        _,N,_,_,_ =selected_images.shape
        
        new_features = [features,selected_images[:,-1,:,:,:]]
        new_features = torch.stack(features,new_features)
        
        for i in range(N-1):
            image = selected_images[:,i,:,:,:]
            new_img = self.AttackMethod(image,new_features)
            new_images.append(new_img)
        new_images = torch.stack(new_images)
        new_images = new_images.permute(1,0,2,3,4)

        return new_images
        

