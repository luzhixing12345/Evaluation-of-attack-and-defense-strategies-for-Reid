

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from fastreid.engine import DefaultTrainer
from fastreid.modeling.heads.build import build_heads
from fastreid.utils.checkpoint import Checkpointer
from advertorch.context import ctx_noparamgrad_and_eval
from fastreid.utils.reid_patch import change_preprocess_image, get_result, get_train_set
device='cuda'
def adversarial_defense_C(cfg,train_set):
    
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


def adversarial_defense_FNA(cfg,train_set):
    
    cfg = DefaultTrainer.auto_scale_hyperparams(cfg,train_set.dataset.num_classes)
    model = DefaultTrainer.build_model_main(cfg)#this model was used for later evaluations
    model.heads = build_heads(cfg)
    model.to(device)
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)


    optimizer = DefaultTrainer.build_optimizer(cfg, model)
    max_id = 100   # the max trainging epoch ,since training set has so many groups, 4000 is a proper choice


    ids = []
    features = []
    model.eval()
    for id, data in enumerate(train_set):
        if id>max_id:
            break
        with torch.no_grad():
            output = model(data['images'].to(device))
        features.append(output)
        ids.append(data['targets'].cpu())
    ids = torch.cat(ids, 0).numpy()
    features = torch.cat(features, 0).cpu()

    guide_features = []

    dict_f = defaultdict(list)
    for f, id in zip(features, ids):
        dict_f[id].append(f)
    
    max_n_id = 0
    for _, values in dict_f.items():
        if len(values) > max_n_id:
            max_n_id = len(values)

    # ## Max distance cluster center
    proto_f = []
    for id, values in dict_f.items():
        proto_f.append(torch.mean(torch.stack(values), dim=0))
    proto_f = torch.stack(proto_f)

    keys = list(dict_f.keys()) # keys[i] = id of proto_f[i]
    # max_dist = True

    for nb_q in range(len(features)):
        id = ids[nb_q]
        guide = []
        for f in dict_f[id]:
            guide.append(f)
        while len(guide) < max_n_id:
            guide.append(features[nb_q])

        id_f = ids[nb_q] # id current feature
        r = np.random.randint(len(proto_f))
        while keys[r] == id_f: # while proto_f[r] has same id as f
            r = np.random.randint(len(proto_f))
        guide.append(proto_f[r])
        guide_features.append(torch.stack(guide)) 

    guide_features = torch.stack(guide_features)
    b_s = train_set.batch_sampler.batch_size
    

    alpha = 0.5     # mixing ratio between clean data and attack data,and it represents the ratio of clean data 
    
    adversary=match_attack_method(cfg,model,train_set)
    adversary.clip_max=255.0
    model.train()
    j=0
    for guides, data in zip(torch.split(guide_features, b_s), train_set):
        if j>max_id:
            break
        j=j+1
        clean_data = data['images'].to(device)
        targets = data['targets'].to(device)
        with ctx_noparamgrad_and_eval(model):
            adv_data = adversary.perturb(clean_data, guides.to(device))
        optimizer.zero_grad()

        output_clean = model(clean_data)
        output_att = model(adv_data)
        
        loss_dict_att = model.losses(output_att, targets)
        loss_dict_clean = model.losses(output_clean,targets)
        
        losses_att = sum(loss_dict_att.values())
        losses_clean = sum(loss_dict_clean.values())
        loss = alpha*losses_clean+(1-alpha)*losses_att
        loss.backward()
        optimizer.step()
        
    print('finished adv_training !')
    print('--------------------------------')
    Checkpointer(model,'model').save('def_trained')
    print('Successfully saved the adv_trained model !')


class adversary_defense:

    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self.train_set = get_train_set(self.cfg)

    def get_defense_result(self):
        return get_result(self.cfg,self.cfg.MODEL.DEFENSE_TRAINED_WEIGHT,'defense')

    def defense(self):
        pass


    