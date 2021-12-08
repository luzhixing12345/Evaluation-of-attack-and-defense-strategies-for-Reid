
#这里引用了github的项目advtorch的库
#如想详细了解请阅读源代码
#https://github.com/BorealisAI/advertorch
from fastreid.utils.reid_patch import evaluate_ssim,get_result,Attack_algorithm_library,change_preprocess_image
from fastreid.modeling.heads import build_feature_heads
from .attack_algorithm import *
from .attack_type import *
from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer

device = 'cuda'
attack_type = ['QA','GA']
# C_Attack_algorithm_library=["FGSM",'IFGSM','MIFGSM','CW','ODFA']  #针对分类问题的攻击算法库
# #DDNL2 SPA CW时间太长了！！！
# R_Attack_algorithm_library=['SMA','FNA','MIS-RANKING']
# Attack_algorithm_library=C_Attack_algorithm_library+R_Attack_algorithm_library

def attack(cfg):
    if cfg.ATTACK_C:
        CA = ClassificationAttack(cfg)
        CA.attack()
        return CA.get_result()
    
    else :
        if cfg.ATTACKTYPE=='QA':
            QA = QueryAttack(cfg)
            QA.attack()
            return QA.get_result()
        else :
            GA = GalleryAttack(cfg)
            GA.attack()
            return GA.get_result()


    return att_result,att_result_to_save,evaluate_ssim(cfg)



def attack_C(cfg,data_loader,model_path,max_batch_id=-1):
    """
    Arguments:

    
    """
    cfg = DefaultTrainer.auto_scale_hyperparams(cfg,data_loader.dataset.num_classes)
    model = DefaultTrainer.build_model_main(cfg)  # use baseline_train
    model.preprocess_image=change_preprocess_image(cfg) # rerange the input size to [0,1]

    Checkpointer(model).load(model_path)  # load trained model
    assert cfg.ATTACKMETHOD in Attack_algorithm_library,"you need to use attack algorithm in the library, or check your spelling"

    model_rank = DefaultTrainer.build_model_main(cfg)#this model was used for later evaluations
    model_rank.preprocess_image = change_preprocess_image(cfg) 
    model_rank.heads = build_feature_heads(cfg)
    model_rank.to(device)
    Checkpointer(model_rank).load(cfg.MODEL.WEIGHTS)

    # model is used for classification(especially for classification attack)
    # model_rank is used for get features(for GA attack)

    if cfg.ATTACKTYPE =="QA":
        query_attack = QA(cfg,model,model_rank)
        query_attack.attack()
        return get_result(cfg,cfg.MODEL.WEIGHTS,'attack')
    
    else :

        gallery_attack = GA(cfg,model,model_rank)
        gallery_attack.attack()
        return get_result(cfg,cfg.MODEL.WEIGHTS,'attack')
    


def attack_R(cfg, query_data_loader,gallery_data_loader,pos):
    
    
    model = DefaultTrainer.build_model_main(cfg)
    model.preprocess_image = change_preprocess_image(cfg) 
    model.heads = build_feature_heads(cfg)
    model.to(device)
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
    _LEGAL_ATTAKS_ = ('ES', 'QA', 'CA', 'SPQA', 'GTM', 'GTT', 'TMA', 'LTM')

    if cfg.ATTACKMETHOD=='SMA':
        adversary=match_attack_method(cfg,model,query_data_loader)
        SMA(query_data_loader,adversary, model,pos)
    elif cfg.ATTACKMETHOD=='FNA':
        adversary=match_attack_method(cfg,model,query_data_loader)
        FNA(query_data_loader,adversary, model,cfg.RAND,pos)
    elif cfg.ATTACKMETHOD=='MIS-RANKING':
        mis_ranking(cfg,query_data_loader,pos)
    elif cfg.ATTACKMETHOD=='MUAP':
        # pay attention that , if you use this attack method
        # remember to add "TEST.IMS_PER_BATCH 32" at the end of the command line to change the 
        # batch_size of query_set to 32 instead of 64
        # or you may meet CUDA out of memory error!!!
        MUAP(cfg,query_data_loader,model,pos)
    elif cfg.ATTACKMETHOD=='SSAE':
        SSAE_attack(cfg,query_data_loader,pos)

    # elif cfg.ATTACKMETHOD in _LEGAL_ATTAKS_ or cfg.ATTACKMETHOD[:-1] in _LEGAL_ATTAKS_:
    #     ROBRANK(cfg,query_data_loader,gallery_data_loader)

    else :
        raise






