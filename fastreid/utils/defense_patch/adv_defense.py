


import torch.nn as nn
from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer
from advertorch.context import ctx_noparamgrad_and_eval
from fastreid.utils.attack_patch.attack_patch import match_attack_method
from fastreid.utils.reid_patch import change_preprocess_image
device='cuda'
def adversarial_defense(cfg,train_set):
    
    cfg = DefaultTrainer.auto_scale_hyperparams(cfg,train_set.dataset.num_classes)
    model = DefaultTrainer.build_model_main(cfg)#this model was used for later evaluations
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)

    model_att_C = DefaultTrainer.build_model_main(cfg)#this model was used for produce attack images for C_attack
    model_att_C.preprocess_image = change_preprocess_image(cfg)
    Checkpointer(model_att_C).load(cfg.MODEL.WEIGHTS)


    optimizer = DefaultTrainer.build_optimizer(cfg, model)
    adversary=match_attack_method(cfg,model_att_C,train_set)

    alpha = 0.5     # mixing ratio between clean data and attack data,and it represents the ratio of clean data 
    max_id = 4000   # the max trainging epoch ,since training set has so many groups, 4000 is a proper choice
    loss_fun = nn.CrossEntropyLoss()
    model.train()
    for idx ,data in enumerate(train_set):
        if idx>max_id:
            break
        clean_data = data['images'].to(device)
        with ctx_noparamgrad_and_eval(model_att_C):
            adv_data = adversary.perturb(clean_data/255.0)#the data range of adv_data is 0.0~1.0

        targets = data['targets'].to(device)
        optimizer.zero_grad()
        logits_clean = model(clean_data)
        logits_att = model(adv_data*255.0)
        loss = loss_fun(logits_clean,targets)*alpha+loss_fun(logits_att,targets)*(1-alpha)
        loss.backward()
        optimizer.step()
        
    print('finished adv_training !')
    print('--------------------------------')
    Checkpointer(model,'model').save('def_trained')
    print('Successfully saved the adv_trained model !')
