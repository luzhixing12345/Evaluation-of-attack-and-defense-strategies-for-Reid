import torch 
import torch.nn as nn
import torch.nn.functional as F
from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.reid_patch import eval_test, eval_train, get_result, get_train_set
device='cuda'


#https://zhuanlan.zhihu.com/p/31177892?iam=995bd462e83ba25488ec849e8949e1f8
def distillation(cfg,train_data_loader):
    # train a robust model again with another defense machanism

    cfg = DefaultTrainer.auto_scale_hyperparams(cfg,train_data_loader.dataset.num_classes)
    model = DefaultTrainer.build_model_main(cfg)  #启用baseline_for_defense
    model.RESIZE = True
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
    
    #optimizer = optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=1e-5)
    #optimizer = optim.SGD(model.parameters(),lr=0.00001,weight_decay=1e-5) # for bot_r50
    optimizer = DefaultTrainer.build_optimizer(cfg, model)
    
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    T = 100  #distillation temperature
    max_epoch = 1000
    EPOCH = 3
    
    for epoch in range(EPOCH):
        loss_total = 0
        model.train()
        model.heads.MODE = 'C'
        for batch_idx,data in enumerate(train_data_loader):
            if batch_idx>max_epoch:
                break
            clean_data = data['images'].to(device)
            targets = data['targets'].to(device)
                    
            logits = model(clean_data)
            loss = criterion(logits/T,targets)
            
            loss_total+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('loss_total = ',loss_total)
    
    distillation_labels = []
    
    for batch_idx,data in enumerate(train_data_loader):
        if batch_idx>max_epoch:
            break
        clean_data = data['images'].to(device)
        with torch.no_grad():
            outputs = softmax(model(clean_data))
            softlabels = outputs.argmax(dim=1)
        distillation_labels.append(softlabels)
    
    model_dis = DefaultTrainer.build_model_main(cfg)  #启用baseline_for_defense
    Checkpointer(model_dis).load(cfg.MODEL.WEIGHTS)  # load trained model
    optimizer_dis = DefaultTrainer.build_optimizer(cfg, model_dis)
    model_dis.train()
    model_dis.heads.MODE = 'C'
    for epoch in range(EPOCH):
        loss_total = 0
        model_dis.heads.MODE = 'C'
        for batch_idx,data in enumerate(train_data_loader):
            if batch_idx>max_epoch:
                break
            clean_data = data['images'].to(device)
            targets = distillation_labels[batch_idx]
        
            logits = model_dis(clean_data)
            loss = criterion(logits,targets)
            loss_total+=loss.item()
            
            optimizer_dis.zero_grad()
            loss.backward()
            optimizer.step()
        eval_train(model_dis,train_data_loader)
        eval_test(cfg,model,epoch)
        print('loss_total = ',loss_total)
    
    
    print('finished dstillation_training !')
    Checkpointer(model_dis,'model').save(f'{cfg.DEFENSEMETHOD}_{cfg.DATASETS.NAMES[0]}_{cfg.CFGTYPE}')
    print('Successfully saved the distill_trained model !')


class distillation_defense:

    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self.train_set = get_train_set(self.cfg)

    def get_defense_result(self):
        return get_result(self.cfg,self.cfg.MODEL.DEFENSE_TRAINED_WEIGHT,'defense')
  
    def defense(self):
        distillation(self.cfg,self.train_set)
