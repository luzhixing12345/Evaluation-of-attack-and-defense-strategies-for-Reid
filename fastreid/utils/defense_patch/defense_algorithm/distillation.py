import torch 
import torch.nn as nn
from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.reid_patch import change_preprocess_image, eval_train, get_result, get_train_set
device='cuda'


#https://zhuanlan.zhihu.com/p/31177892?iam=995bd462e83ba25488ec849e8949e1f8
def distillation(cfg,train_data_loader):
    # train a robust model again with another defense machanism

    cfg = DefaultTrainer.auto_scale_hyperparams(cfg,train_data_loader.dataset.num_classes)
    model = DefaultTrainer.build_model_main(cfg)  #启用baseline_for_defense
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

    print(eval_train(model,train_data_loader,4000))
    torch.nn.init.xavier_normal_(model.heads.classifier.weight)
    #optimizer = optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=1e-5)
    #optimizer = optim.SGD(model.parameters(),lr=0.00001,weight_decay=1e-5) # for bot_r50
    optimizer = DefaultTrainer.build_optimizer(cfg, model)
    
    loss_fun = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    T = 100  #distillation temperature
    model.train()
    
    for i in range(5):
        loss_total = 0
        for batch_idx,data in enumerate(train_data_loader):
            if batch_idx>4000:
                break
            clean_data = data['images'].to(device)
            targets = data['targets'].to(device)
        
            optimizer.zero_grad()
            logits = model(clean_data)

            loss = loss_fun(logits/T,targets)
            loss_total+=loss.item()
            loss.backward()
            optimizer.step()
        accurency = eval_train(model,train_data_loader,1000)
        print('The accurency of query set in Train Epoch {} is {}%'.format(i,accurency))

        print('loss_total = ',loss_total)
    
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

    print('finished dstillation_training !')
    Checkpointer(model,'model').save('def_trained')
    print('Successfully saved the distill_trained model !')


class distillation_defense:

    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self.train_set = get_train_set(self.cfg)

    def get_defense_result(self):
        return get_result(self.cfg,self.cfg.MODEL.DEFENSE_TRAINED_WEIGHT,'defense')
  
    def defense(self):
        distillation(self.cfg,self.train_set)
