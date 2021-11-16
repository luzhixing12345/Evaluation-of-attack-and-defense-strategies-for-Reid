
import torch
import numpy as np
from collections import defaultdict
from fastreid.utils.reid_patch import process_set,pairwise_distance, save_image


def FNA(q_loader, attack, model,rand,pos,device='cuda'):
    """Perturb the queries with FNA
    
    Arguments:
        :q_loader {pytorch dataloader} -- dataloader of the query dataset
        :attack {advertorch.attack} -- adversarial attack to perform on the queries
        :model {pytorch model} -- pytorch model to evaluate
        :device {cuda device} --
        :cosine {bool} -- evaluate with cosine similarity
        :triplet {bool} -- model trained with triplet or not
        :classif {bool} -- model trained with cross entropy
        :transforms {bool} -- apply transforms
    
    Returns:
        :features -- Tensor of the features of the queries
        :ids -- numpy array of ids
        :cams -- numpy array of camera ids
    """
    model.eval()
    probe_features, probe_ids, probe_cams = process_set(q_loader, model)

    guide_features = []

    dict_f = defaultdict(list)
    for f, id in zip(probe_features, probe_ids):
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

    # if not cosine:
    dist_proto = pairwise_distance(probe_features, proto_f)
    keys = list(dict_f.keys()) # keys[i] = id of proto_f[i]
    # max_dist = True

    for nb_q in range(len(probe_features)):
        q_d_proto = dist_proto[nb_q]
        id = probe_ids[nb_q]
        guide = []
        for f in dict_f[id]:
            guide.append(f)
        while len(guide) < max_n_id:
            guide.append(probe_features[nb_q])

        if rand: # Choosing a random cluster different from own cluster
            id_f = probe_ids[nb_q] # id current feature
            r = np.random.randint(len(proto_f))
            while keys[r] == id_f: # while proto_f[r] has same id as f
                r = np.random.randint(len(proto_f))
            guide.append(proto_f[r])
        else: # Choosing furthest cluster
            max_d_p = -np.inf
            max_ind_p = 0
            for i,d_p in enumerate(q_d_proto):
                if d_p > max_d_p and probe_ids[nb_q] != keys[i]:
                    max_d_p = d_p
                    max_ind_p = i
            guide.append(proto_f[max_ind_p])
        guide_features.append(torch.stack(guide)) 

    guide_features = torch.stack(guide_features)
    b_s = q_loader.batch_sampler.batch_size
    for guides, data in zip(torch.split(guide_features, b_s), q_loader):
        image_adv = attack.perturb((data['images']/255).to(device), guides.to(device))
        path = data['img_paths']
        save_image(image_adv,path,pos)