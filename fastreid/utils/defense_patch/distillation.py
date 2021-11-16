import torch 
import torch.nn as nn
from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.reid_patch import eval_train
device='cuda'


#https://zhuanlan.zhihu.com/p/31177892?iam=995bd462e83ba25488ec849e8949e1f8
def distillation(cfg,train_data_loader):
    # train a robust model again with another defense machanism

    cfg = DefaultTrainer.auto_scale_hyperparams(cfg,train_data_loader.dataset.num_classes)
    model = DefaultTrainer.build_model_main(cfg)  #启用baseline_for_defense
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
    print(eval_train(model,train_data_loader,400))
    #optimizer = optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=1e-5)
    #optimizer = optim.SGD(model.parameters(),lr=0.00001,weight_decay=1e-5) # for bot_r50
    optimizer = DefaultTrainer.build_optimizer(cfg, model)
    
    loss_fun = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

    max_id = 2000
    T = 5  #distillation temperature
    distillation_targets = []

    model.train()
    
    for i in range(5):
        for batch_idx,data in enumerate(train_data_loader):
            if batch_idx>max_id :
                break
            clean_data = data['images'].to(device)
            targets = data['targets'].to(device)
            optimizer.zero_grad()

            logits = model(clean_data)

            if i==4:
                probabilities = softmax(logits/T)
                new_targets = probabilities.argmax(dim=1,keepdim=False)
                distillation_targets.append(new_targets)

            loss = loss_fun(logits,targets)
            loss.backward()
            optimizer.step()
        print(eval_train(model,train_data_loader,400),i)
    
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
    model.train()
    for i in range(3):
        for batch_idx,data in enumerate(train_data_loader):
            if batch_idx>max_id :
                break
            clean_data = data['images'].to(device)
            targets = distillation_targets[batch_idx].to(device)
            optimizer.zero_grad()

            logits = model(clean_data)
            probabilities = softmax(logits/T)

            loss = loss_fun(probabilities,targets)
            loss.backward()
            optimizer.step()
        
    print('finished dstillation_training !')
    Checkpointer(model,'model').save('def_trained')
    print('Successfully saved the distill_trained model !')