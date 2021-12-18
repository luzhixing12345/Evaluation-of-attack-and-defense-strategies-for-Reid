
import torch
import torch.nn as nn

from fastreid.utils.reid_patch import log

from .utils import Attack,LabelMixin
from advertorch.utils import clamp
from advertorch.utils import batch_multiply



class GradientSignAttack(Attack, LabelMixin):
    """
    One step fast gradient sign method (Goodfellow et al, 2014).
    Paper: https://arxiv.org/abs/1412.6572

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: attack step size.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: indicate if this is a targeted attack.
    """

    def __init__(self, cfg, predict, loss_fn=None, eps=0.3, clip_min=0.,
                 clip_max=1., targeted=False):
        """
        Create an instance of the GradientSignAttack.
        """
        super(GradientSignAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.cfg = cfg
        self.eps = eps
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        if(self.cfg.ATTACKTYPE=='GA'):
            return self.perturb_GA(x,y)

        x, y = self._verify_and_process_inputs(x, y)
        xadv = x.requires_grad_()

        logits = self.predict(xadv)
        loss = self.loss_fn(logits,y)
        if self.targeted:
            loss = -loss

        loss.backward()
        grad_sign = xadv.grad.detach().sign()

        xadv = xadv + batch_multiply(self.eps, grad_sign)
 
        xadv = clamp(xadv, self.clip_min, self.clip_max)

        return xadv.detach()

    def perturb_GA(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """

        _,N,_,_,_=x.shape

        new_x=[]
        for i in range(N):
            xadv = x[:,i,:,:,:].clone().detach()
            xadv.requires_grad_()
            logits = self.predict(xadv)
            loss = self.loss_fn(logits,y)

            if self.targeted:
                loss = -loss

            loss.backward()
            grad_sign = xadv.grad.detach().sign()

            xadv = xadv + batch_multiply(self.eps, grad_sign)
 
            xadv = clamp(xadv, self.clip_min, self.clip_max)
            new_x.append(xadv)

        new_x = torch.stack(new_x)
        new_x = new_x.permute(1,0,2,3,4)
        return new_x


FGSM = GradientSignAttack


