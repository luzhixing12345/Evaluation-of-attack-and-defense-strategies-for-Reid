
import torch.nn as nn
from advertorch.defenses import MedianSmoothing2D
from advertorch.defenses import BitSqueezing
from advertorch.defenses import JPEGFilter
from fastreid.utils.attack_patch.attack_patch import match_attack_method
from fastreid.utils.reid_patch import save_image

device='cuda'
def smooth(cfg,model,query_set):
    defense_set = nn.Sequential(
                                BitSqueezing(bit_depth=5),
                                MedianSmoothing2D(kernel_size=3),
                                JPEGFilter(10)
                                )

    adversary = match_attack_method(cfg,model,query_set)
    for data in query_set:
        clip_data, true_label, path = data['images'].to(device),data['targets'].to(device),data['img_paths'].to(device)      
        adv_data = adversary.perturb(clip_data,true_label)
        adv_defended=defense_set(adv_data)
        save_image(adv_defended,path,'def_adv_query')