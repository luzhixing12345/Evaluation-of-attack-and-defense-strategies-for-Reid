

from fastreid.utils.attack_patch.attack_patch import attack
from fastreid.utils.reid_patch import get_result,release_cuda_memory
from fastreid.utils.defense_patch.adv_defense import adversarial_defense
from fastreid.utils.defense_patch.gra_regulation import gradient_regulation
from fastreid.utils.defense_patch.goat import GOAT



device = 'cuda'
# G_Defense_algorithm_library=['ADV_DEF','GRA_REG']
# R_Defense_algorithm_library=['GOAT']
# Defense_algorithm_library=G_Defense_algorithm_library+R_Defense_algorithm_library

def defense(cfg,train_set,query_set,gallery_set,type:bool):

    if type:
        defense_G(cfg,train_set)
    else :
        defense_R(cfg,train_set,query_set,gallery_set)
        
    def_result= get_result(cfg,cfg.MODEL.DEFENSE_TRAINED_WEIGHT,'defense')
    def_adv_result= get_result(cfg,cfg.MODEL.DEFENSE_TRAINED_WEIGHT,'attack')

    return def_result,def_adv_result
        
def defense_G(cfg,train_set):
    if cfg.MODEL.DEFENSEMETHOD=='ADV_DEF':
        adversarial_defense(cfg,train_set)
        
    elif cfg.MODEL.DEFENSEMETHOD=='GRA_REG':
        gradient_regulation(cfg,train_set)
    
    release_cuda_memory()


def defense_R(cfg,train_set,query_set,gallery_set):
    if cfg.MODEL.DEFENSEMETHOD=='GOAT':
        GOAT(cfg,train_set,pull=cfg.PULL,nb_r=cfg.NB_R)
    else :
        raise 

    release_cuda_memory()