

import torch.nn as nn
from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.reid_patch import get_train_set
import torch
import torch.nn as nn
import torch.autograd as autograd

from torch import Tensor
device='cuda'
def gradient_regulation(cfg,train_data_loader):
    # train a robust model again with another defense machanism

    train_cfg = DefaultTrainer.auto_scale_hyperparams(cfg,train_data_loader.dataset.num_classes)

    model = DefaultTrainer.build_model_main(train_cfg)  #启用baseline_for_defense
    Checkpointer(model).load(train_cfg.MODEL.WEIGHTS)  # load trained model
    
    #optimizer = optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=1e-5)
    #optimizer = optim.SGD(model.parameters(),lr=0.00001,weight_decay=1e-5) # for bot_r50
    optimizer = DefaultTrainer.build_optimizer(train_cfg, model)
    
    loss_fun = nn.CrossEntropyLoss()
    loss_calcuation = InputGradRegLoss(weight = 500.0,criterion = loss_fun,norm = 'L2')

    max_id = 4000
    
    model.train()
    
    for batch_idx,data in enumerate(train_data_loader):
        if batch_idx>max_id :
            break
        clean_data = data['images']
        targets = data['targets'].to(device)
        optimizer.zero_grad()
        clean_data = clean_data.requires_grad_().to(device)
        model.eval()
        logits = model(clean_data)
        model.train()
        loss = loss_calcuation(logits,targets,clean_data) # weight的值还要好好选择一下
        loss.backward()
        optimizer.step()


    print('finished gra_training !')
    Checkpointer(model,'model').save('def_trained')
    print('Successfully saved the gra_trained model !')





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