
import torch
import torch.nn as nn

from fastreid.engine import DefaultTrainer
from fastreid.modeling.heads.build import build_feature_heads
from fastreid.utils.checkpoint import Checkpointer
from collections import defaultdict

from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.advertorch.attacks import PGDAttack
from fastreid.utils.reid_patch import change_preprocess_image, pairwise_distance
import numpy as np
from torchvision import transforms, datasets

device='cuda'

#Rank 防御方法
def GOAT(cfg,train_data_loader,pull,nb_r):

    cfg = DefaultTrainer.auto_scale_hyperparams(cfg,train_data_loader.dataset.num_classes)
    model = DefaultTrainer.build_model(cfg)
    preprocess_image = model.preprocess_image
    model.preprocess_image = change_preprocess_image(cfg)
    model.to(device)
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)

    optimizer = DefaultTrainer.build_optimizer(cfg, model)
    max_epoch = 1000
    request_dict = defaultdict(list)
    i=0
    for data in train_data_loader:
        if i>max_epoch:
            break
        i=i+1
        targets =data['targets'].cpu()
        image = data['images'].cpu()
        for id,img in zip(targets,image):
            request_dict[id.item()].append(img)

    
    for index, data in enumerate(train_data_loader):
        if index>max_epoch:
            break
        inputs_clean = (data['images']/255.0).to(device)  #64*3*256*128
        labels = data['targets'].to(device)        #64
            
        inputs = create_adv_batch(model,inputs_clean,labels.cpu(),request_dict,pull=pull,nb_r=nb_r)

        model.train()
        # zero the parameter gradients
        optimizer.zero_grad()

        input_data={'images':inputs.to(device),'targets':labels}
        loss_dict = model(input_data)
        losses = sum(loss_dict.values())

        losses.backward()
        optimizer.step()
    
    model.preprocess_image = preprocess_image
    print('finished GOAT_training !')
    Checkpointer(model,'model').save('def_trained')
    print('Successfully saved the GOAT_trained model !')
 
def create_adv_batch(model,inputs,labels,request_dict,pull=True,nb_r=1):
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
    requests = []
    if nb_r > 1:
        for i in range(nb_r):
            R = []
            for label in labels:
                image = request_dict[label.item()]
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


    attack = PGDAttack(lambda x: model(x), criterion, eps=0.05, nb_iter=7, eps_iter=1.0/255.0, ord=np.inf,rand_init=True)
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