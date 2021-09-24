

import torch.nn as nn
from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.reid_patch import train_train_set
from fastreid.utils.attack_patch.attack_patch import match_attack_method
from advertorch.context import ctx_noparamgrad_and_eval
device='cuda'
def adversarial_defense(cfg):
    
    train_data_loader = DefaultTrainer.build_train_loader(cfg) #读取训练集
    train_train_set(cfg,train_data_loader)#训练training set的分类层，建立图像到id的映射
    train_cfg = DefaultTrainer.auto_scale_hyperparams(cfg,train_data_loader.dataset.num_classes)
    model = DefaultTrainer.build_model_for_attack(train_cfg)
    Checkpointer(model).load(cfg.MODEL.TRAINSET_TRAINED_WEIGHT)

    optimizer = DefaultTrainer.build_optimizer(train_cfg, model)
    adversary = match_attack_method(train_cfg,model,train_data_loader)#对抗性攻击，找到所对抗的攻击算法

    alpha = 0.8   #混合比例
    epoch = 4000 #4000
    loss_fun = nn.CrossEntropyLoss()
    model.train()
    for batch_idx,data in enumerate(train_data_loader):
        if batch_idx>epoch:
            break
        clean_data = data['images']
        targets = data['targets'].to(device)
        with ctx_noparamgrad_and_eval(model):
            adv_data = adversary.perturb(clean_data)
        optimizer.zero_grad()
        logits_adv = model(adv_data)
        logits_clean = model(clean_data)
        loss = alpha*loss_fun(logits_clean,targets)+(1-alpha)*loss_fun(logits_adv,targets)
        loss.backward()
        optimizer.step()
        if batch_idx%400==0:
            print(f'the training for the training set has finished the {batch_idx} / epoch ')
            print('------------------------------------------------------------------------')
    print('finished adv_training !')
    print('--------------------------------')
    Checkpointer(model,'model').save('adv_trained')
    print('Successfully saved the adv_trained model !')
