

from fastreid.utils.reid_patch import get_result, match_type,release_cuda_memory
from fastreid.utils.defense_patch.adv_defense import adversarial_defense_C,adversarial_defense_SMA,adversarial_defense_FNA
from fastreid.utils.defense_patch.gra_regulation import gradient_regulation
from fastreid.utils.defense_patch.goat import GOAT
from fastreid.utils.defense_patch.distillation import distillation
from fastreid.utils.defense_patch.robrank_defense import robrank_defense


device = 'cuda'
# G_Defense_algorithm_library=['ADV_DEF','GRA_REG']
# R_Defense_algorithm_library=['GOAT']
# Defense_algorithm_library=G_Defense_algorithm_library+R_Defense_algorithm_library

def defense(cfg,train_set,query_set,gallery_set,type:bool):

    if type:
        defense_G(cfg,train_set)
    else :
        defense_R(cfg,train_set,query_set,gallery_set)
        
    def_result,def_result_to_save= get_result(cfg,cfg.MODEL.DEFENSE_TRAINED_WEIGHT,'defense')
    def_adv_result,def_adv_result_to_save= get_result(cfg,cfg.MODEL.DEFENSE_TRAINED_WEIGHT,'attack')

    return def_result,def_result_to_save,def_adv_result,def_adv_result_to_save
        
def defense_G(cfg,train_set):
    if cfg.DEFENSEMETHOD=='ADV_DEF':
        if match_type(cfg,'attack'):
            adversarial_defense_C(cfg,train_set)
        else :
            if cfg.MODEL.ATTACKMETHOD=='SMA':
                adversarial_defense_SMA(cfg,train_set)
            elif cfg.MODEL.ATTACKMETHOD=='FNA':
                adversarial_defense_FNA(cfg,train_set)
            else:
                pass
        
    elif cfg.DEFENSEMETHOD=='GRA_REG':
        gradient_regulation(cfg,train_set)
    
    elif cfg.DEFENSEMETHOD=='DISTILL':
        distillation(cfg,train_set)
        
    release_cuda_memory()


def defense_R(cfg,train_set,query_set,gallery_set):
    _robrank_set_ =['SES','EST','PNP']
    if cfg.DEFENSEMETHOD=='GOAT':
        GOAT(cfg,train_set,pull=cfg.PULL,nb_r=cfg.NB_R)
    elif cfg.DEFENSEMETHOD in _robrank_set_:
        robrank_defense(cfg,train_set,cfg.DEFENSEMETHOD)
    else :
        raise 

    release_cuda_memory()