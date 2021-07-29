
"""
Implementation of regularization in 
Ross et al.,
"Improving the Adversarial Robustness and Interpretability of Deep Neural Networks by Regularizing their Input Gradients",
AAAI 18'.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
import torch.autograd as autograd

from torch import Tensor


class InputGradReg(object):
    def __init__(self, norm="L1"):
        super(InputGradReg, self).__init__()
        self.norm = norm.upper()

    def __call__(self, loss: Tensor, x: Tensor, y:Tensor=None):
        if x.requires_grad == False:
            raise ValueError("Input must enable requires_grad")

        if loss.ndimension() != 0:
            grad_outputs = torch.ones_like(loss, dtype=loss.dtype, device=loss.device)
        else:
            grad_outputs = None

        # .grad() return a tuple
        # grad_outputs 是一个shape 与 outputs 一致的向量
        # loss只能为标量，但若loss输入为向量的情况，grad_outputs内部和其点乘求和，在对x的每个分量分别求导数
        # 知乎：https://zhuanlan.zhihu.com/p/83172023
        # 简单地理解成在求梯度时的权重
        x_grad = autograd.grad(loss, x, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]

        if self.norm == "L1":
            regulr = torch.sum(torch.abs(x_grad)) # torch.norm(x_grad, p=self.norm, dim=1)
        elif self.norm == "L2":
            regulr = torch.sum(torch.pow(x_grad, 2))
        else:
            raise NotImplementedError("norm=\"L1\" or norm=\"L2\"")

        return regulr

class InputGradRegLoss(object):
    def __init__(self, weight, criterion=None, norm="L1"):
        super(InputGradRegLoss, self).__init__()

        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        
        self.reg = InputGradReg(norm)
        self.weight = weight
    
    def __call__(self, pred, target, x):
        loss = self.criterion(pred, target)
        regulr = self.reg(loss, x, target)
        return loss + regulr * self.weight
