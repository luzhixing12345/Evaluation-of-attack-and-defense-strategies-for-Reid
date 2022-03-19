


import torch
from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer
import torch.nn as nn
from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.advertorch.attacks import PGDAttack
from fastreid.utils.reid_patch import eval_ssim, eval_test, eval_train, get_result, get_train_set, make_dict, pairwise_distance
import numpy as np
import time
from .sort_dataset import sort_datasets
device='cuda'

#Rank 防御方法
def GOAT(cfg,train_data_loader):

    cfg = DefaultTrainer.auto_scale_hyperparams(cfg,train_data_loader.dataset.num_classes)
    model = DefaultTrainer.build_model_main(cfg)
    
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    model.to(device)

    optimizer = DefaultTrainer.build_optimizer(cfg, model)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.00035, weight_decay=5e-4)

    max_epoch = 2000
    EPOCH = 5
    frequency = 800
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.MarginRankingLoss(margin=10, reduction='mean')
    
    # model.train()
    # for _,parm in enumerate(model.parameters()):
    #     parm.requires_grad=True
    # print('all parameters of model requires_grad')
    request_dict = sort_datasets(cfg.DATASETS.NAMES[0])
    # train_batch_size = cfg.SOLVER.IMS_PER_BATCH
    # N = 18000//train_batch_size
    # for idx,data in enumerate(train_data_loader):
    #     if idx>N:
    #         break
    #     images = data['images'].cpu()
    #     labels = data['targets'].cpu()
    #     for img,label in zip(images,labels):
    #         request_dict[label.item()].append(img)
        
    #     if idx%100==0:
    #         print(len(request_dict.keys()))
    
    for epoch in range(EPOCH):
        loss_total = 0
        print(f'start training for epoch {epoch} of {EPOCH}')
        time_stamp_start = time.strftime("%H:%M:%S", time.localtime()) 
        model.train()
        for index, data in enumerate(train_data_loader):
            if index>max_epoch:
                break
            
            inputs_clean = data['images'].to(device)
            labels = data['targets'].to(device)       
                
            model.heads.MODE = 'F'
            adv_images = create_adv_batch(model,inputs_clean,labels.cpu(),request_dict)
            if index % frequency ==0:
                print('ssim = ',eval_ssim(inputs_clean,adv_images))

            model.heads.MODE = 'C'
            # zero the parameter gradients
            adv_images = adv_images.clone().detach()
            adv_images.requires_grad_()
            optimizer.zero_grad()
            outputs = model(adv_images)
            #loss_dict = model.losses(outputs,labels)
            #loss = sum(loss_dict.values())
            # dist = pairwise_distance(outputs, outputs)
            # dist_ap, dist_an, list_triplet = get_distances(dist, labels)
            # y = torch.ones(dist_ap.size(0)).to(device)
            # loss = criterion(dist_an, dist_ap, y)
            loss = criterion(outputs,labels)
            loss_total+=loss.item()
                
            loss.backward()
            optimizer.step()   
        eval_train(model,train_data_loader)
        eval_test(cfg,model,epoch)
        time_stamp_end = time.strftime("%H:%M:%S", time.localtime()) 
        print(f'total_loss for epoch {epoch} of {EPOCH} is {loss_total} | {time_stamp_start} - {time_stamp_end}')

    print('finished GOAT_training !')
    Checkpointer(model,'model').save(f'{cfg.DEFENSEMETHOD}_{cfg.DATASETS.NAMES[0]}_{cfg.CFGTYPE}')
    print(f'Successfully saved the {cfg.DEFENSEMETHOD}_{cfg.DATASETS.NAMES[0]}_{cfg.CFGTYPE} model !')
 

def create_adv_batch(model,inputs,labels,request_dict,rand_r=True,pull=True,nb_r=4):
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
    model.eval()
    model.heads.MODE = 'F'
    if request_dict: # GOAT : INTER BATCH
        requests = []
        if nb_r > 1:
            for i in range(nb_r):
                R = []
                for label in labels:
                    image = request_dict[label.item()]
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
                image = request_dict[label.item()]
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

    attack = PGDAttack(lambda x: model(x), criterion, eps=5/255, nb_iter=7, eps_iter=1.0/255.0, ord=np.inf,rand_init=True)
    data_adv = attack.perturb(inputs, requests)
    return data_adv

mse = torch.nn.MSELoss(reduction='sum')

def get_distances(dist, ids):
    """
    Compute the largest positive distance and smallest negative distance for 
    each element as anchor and returns the batch of positive distance and 
    negative distance.
    Args:
        dist (Tensor): Matrix of the pairwise distances of the batch.
        ids (list): List of the ids of each batch instance.
    Returns:
        batch of largest positive distance
        batch of smallest negative distance
        list of triplets
    """
    dist_an = []
    dist_ap = []
    list_triplet = []
    for index_i, id_i in enumerate(ids):
        max_pos_dist = 0
        pos_pair = (0,0)
        min_neg_dist = np.inf
        neg_pair = (0,0)
        for index_j, id_j in enumerate(ids):
            if index_j == index_i:
                continue
            if id_i == id_j:
                if dist[index_i][index_j] > max_pos_dist:
                    max_pos_dist = dist[index_i][index_j]
                    pos_pair = (index_i, index_j)
            else:
                if dist[index_i][index_j] < min_neg_dist:
                    min_neg_dist = dist[index_i][index_j]
                    neg_pair = (index_i, index_j)
        dist_ap.append(dist[pos_pair[0]][pos_pair[1]])
        dist_an.append(dist[neg_pair[0]][neg_pair[1]])
        list_triplet.append((pos_pair[0],pos_pair[1],neg_pair[1]))
    dist_ap = torch.stack(dist_ap)
    dist_an = torch.stack(dist_an)
    return dist_ap, dist_an, list_triplet

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

class goat_defense:
    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self.train_set = get_train_set(self.cfg)

    def defense(self):
        # requests = sort_datasets(self.cfg.DATASETS.NAMES[0])
        print('\n-----start GOAT defensing-----\n')
        GOAT(self.cfg,self.train_set)

    def get_defense_result(self):
        return get_result(self.cfg,self.cfg.MODEL.DEFENSE_TRAINED_WEIGHT,'defense')

