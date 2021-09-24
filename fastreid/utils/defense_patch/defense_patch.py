
from fastreid.utils.reid_patch import evaluate_misMatch,pairwise_distance
from fastreid.utils.attack_patch.attack_patch import attack_C,attack_R
from fastreid.utils.defense_patch.adv_defense import adversarial_defense
from fastreid.utils.defense_patch.gra_regulation import gradient_regulation
import torch.nn as nn
import torch
from collections import defaultdict

from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.advertorch.attacks import PGDAttack
import numpy as np

device = 'cuda'
Defense_algorithm_library=['ADV_DEF','GRA_REG','GOAT']


def defense(cfg,query_set):
    assert cfg.MODEL.DEFENSEMETHOD in Defense_algorithm_library,"you need to use defend algorithm in the library, or check your spelling"
    if cfg.MODEL.DEFENSEMETHOD=='SMOOTH2D':
        pass
        #smooth(cfg,model,query_set)
        #DefaultTrainer.def_advtest(cfg, model) 
    elif cfg.MODEL.DEFENSEMETHOD=='ADV_DEF':
        adversarial_defense(cfg)
        
    elif cfg.MODEL.DEFENSEMETHOD=='GRA_REG':
        gradient_regulation(cfg)

    elif cfg.MODEL.DEFENSEMETHOD=='GOAT':
        GOAT(cfg,query_set)
    query_cfg = DefaultTrainer.auto_scale_hyperparams(cfg,query_set.dataset.num_classes)
    def_model = DefaultTrainer.build_model_for_attack(query_cfg) 
    Checkpointer(def_model).load(query_cfg.MODEL.DEFENSE_TRAINED_WEIGHT )  # load adv_trained model
    def_misMatch=evaluate_misMatch(def_model,1)
    def_result= DefaultTrainer.test(query_cfg, def_model)
    def_result['misMatch']=def_misMatch
    attack_C(query_cfg,def_model,query_set,pos='def_adv_query')
    def_adv_result =DefaultTrainer.def_advtest(query_cfg,def_model)
    def_adv_misMatch  = evaluate_misMatch(def_model,query_set)
    def_adv_result['misMatch']=def_adv_misMatch
    return def_result,def_adv_result
        
    # else :
    #     print('???')


    


#Rank 防御方法
def GOAT(cfg):
    train_data_loader = DefaultTrainer.build_train_loader(cfg) #读取训练集
    train_cfg = DefaultTrainer.auto_scale_hyperparams(cfg,train_data_loader.dataset.num_classes)
    model = DefaultTrainer.build_model_for_attack(train_cfg)
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)

    optimizer = DefaultTrainer.build_optimizer(train_cfg, model)
    criterion = nn.CrossEntropyLoss()
    idx = defaultdict(list)
    for id,data in enumerate(train_data_loader):
        if id>4000:
            break
        label=data['targets'].to(device)
        image=data['images'].to(device)
        idx[label].append(image)
    model.train()

    for index, batch in enumerate(train_data_loader):
        if index>4000:
            break
        inputs_clean = batch['images'].float().to(device)
        labels = batch['targets'].to(device)
            
        inputs = create_adv_batch(model, inputs_clean, labels, device, rand_r=True, nb_r=4, request_dict=idx, pull=True, self_sufficient=False)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels.squeeze())

        loss.backward()
        optimizer.step()
    
    print('finished GOAT_training !')
    Checkpointer(model,'model').save('adv_trained')
    print('Successfully saved the GOAT_trained model !')
 
def create_adv_batch(model, inputs, labels, device, epsilon=5.0, momentum=1.0, rand_r=True, classif=False, triplet=True, request_dict=None, self_sufficient=False, transforms=True, nb_r=1, pull=False):
    """Create an adversarial batch from a given batch for adversarial training with GOAT

    Args:
        model (nn.Module): model that will serve to create the adversarial images
        inputs (Tensor): batch of original images
        labels (Tensor): labels of the original images
        device (cuda Device): cuda device
        epsilon (int, optional): adversarial perturbation budget. Defaults to 5.0.
        momentum (float, optional): momentum for the attack. Defaults to 1.0.
        rand_r (bool, optional): whether the guides are chosen at random. Defaults to True.
        classif (bool, optional): whether the models is trained by cross entropy or not. Defaults to False.
        triplet (bool, optional): whether the models is trained with triplet loss. Defaults to True.
        request_dict (dict, optional): dictionary that contains the possible guides. Defaults to None.
        self_sufficient (bool, optional): whether the attack is self sufficient (use an artificial guide). Defaults to False.
        transforms (bool, optional): whether to apply the transformation or not. Defaults to True.
        nb_r (int, optional): number of guides. Defaults to 1.
        pull (bool, optional): whether to sample pulling guides. Defaults to False.

    Returns:
        Tensor: batch of adversarial images
    """
    
    if request_dict: # GOAT : INTER BATCH
        requests = []
        if nb_r > 1:
            for i in range(nb_r):
                R = []
                for label in labels:
                    image = request_dict[label.cpu().numpy()[0]]
                    if rand_r:
                        r = np.random.randint(len(image))
                    else:
                        r = i%(len(image))
                    R.append(image[r])
                R = torch.stack(R).to(device)
                with torch.no_grad():
                    R = model(R)
                requests.append(R)
            requests = torch.stack(requests)
            criterion = sum_mse
        else:
            for label in labels:
                image = request_dict[label.cpu().numpy()[0]]
                if rand_r:
                    r = np.random.randint(len(image))
                else:
                    r = 0
                requests.append(image[r])
            requests = torch.stack(requests)
            criterion = mse
            with torch.no_grad():
                # requests = prediction(model, requests, triplet=triplet, classif=classif)
                requests = model(requests)
        if pull: # GOAT FNA : INTER BATCH
            # FURTHEST INTRA
            with torch.no_grad():
                features = model(inputs) 
            dist_intra = pairwise_distance(features, features) # pairwise distance between features in batch
            pulling = []
            for nb_f in range(len(features)):
                dist_f = dist_intra[nb_f] # list of distances to feature nb_f
                # find max distance in batch
                max_d_p = -np.inf # max distance
                max_ind_p = 0 # max index
                for i,d_p in enumerate(dist_f):
                    if d_p > max_d_p and i != nb_f:
                        max_d_p = d_p 
                        max_ind_p = i
                pulling.append(features[max_ind_p])
            pulling = torch.stack(pulling)
            # add pulling feature at the end of pushing guides -> single pulling guide
            requests = torch.cat((requests,pulling.unsqueeze(0)))
            criterion = sum_dif_mse
        requests = requests.to(device)
    
    max_iter = 7
    attack = PGDAttack(lambda x: model(x), criterion, eps=epsilon/255.0, nb_iter=max_iter, eps_iter=1.0/255.0, ord=np.inf, clip_max=255.0, rand_init=True)
    data_adv = attack.perturb(inputs, requests)
    return data_adv

mse = torch.nn.MSELoss(reduction='sum')

def sum_mse(b,f):
    tmp = 0
    for x in f:
        tmp += mse(b,x)
    return tmp
def sum_dif_mse(b,f):
    tmp = 0
    for i in range(len(f)-1):
        tmp += mse(b,f[i])
    tmp -= mse(b,f[-1])
    return tmp