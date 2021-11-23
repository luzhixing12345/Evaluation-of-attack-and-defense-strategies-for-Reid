
#这里引用了github的项目advtorch的库
#如想详细了解请阅读源代码
#https://github.com/BorealisAI/advertorch
from fastreid.utils.reid_patch import evaluate_ssim, save_image,get_result,classify_test_set,Attack_algorithm_library,change_preprocess_image
from fastreid.modeling.heads import build_feature_heads
from fastreid.utils.advertorch.attacks import FGSM,MomentumIterativeAttack,LinfPGDAttack,CarliniWagnerL2Attack
import torch
from advertorch.context import ctx_noparamgrad_and_eval
from fastreid.utils.attack_patch.SMA import SMA
from fastreid.utils.attack_patch.FNA import FNA
from fastreid.utils.attack_patch.MIS_RANKING import mis_ranking
from fastreid.utils.attack_patch.MUAP import MUAP
#from fastreid.utils.attack_patch import cw  
from fastreid.utils.attack_patch.robrank import ROBRANK
from fastreid.utils.attack_patch.SSAE import SSAE_attack
from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer

device = 'cuda'
# C_Attack_algorithm_library=["FGSM",'IFGSM','MIFGSM','CW','ODFA']  #针对分类问题的攻击算法库
# #DDNL2 SPA CW时间太长了！！！
# R_Attack_algorithm_library=['SMA','FNA','MIS-RANKING']
# Attack_algorithm_library=C_Attack_algorithm_library+R_Attack_algorithm_library

def attack(cfg,query_data_loader,gallery_data_loader,type,pos,model_path='./model/test_trained.pth'):
        # test_dataset,num_query = DefaultTrainer.build_test_loader(cfg,dataset_name=cfg.DATASETS.NAMES[0])
    # images = []
    # pids = []
    # camids = []
    if type:  # 针对分类问题的攻击
        classify_test_set(cfg, query_data_loader)
        # 设计分类器，能够完成query_set的image到target的对应,没有完成gallery_set的对应
        # 模型保存的位置在(./model/test_trained.pth)其对query_set上的数据有较高的识别率，但对原train_set识别率下降
        # 测试了训练query_set使其达到较高准确率后对原train_set标签对应的影响,大概准确率降至20%
        # 最后得到两个模型
        #  cfg.MODEL.WEIGHTS has a high accurency in train-set
        #  cfg.MODEL.TESTSET_TRAINED_WEIGHT has a high accurency in query_set
        # the two models both perform bad in opposite set,so do not mis-use .
        print('finished the train for test set ')
        print('---------------------------------------------')
        attack_C(cfg, query_data_loader,model_path,pos=pos)
        att_result,att_result_to_save = get_result(cfg,cfg.MODEL.WEIGHTS,'attack')
    else:  # 针对排序问题的攻击
        attack_R(cfg, query_data_loader,gallery_data_loader,pos=pos)
        att_result,att_result_to_save = get_result(cfg,cfg.MODEL.WEIGHTS,'attack')

    return att_result,att_result_to_save,evaluate_ssim(cfg)



def attack_C(cfg,data_loader,model_path,pos='adv_query',max_batch_id=-1):
    """
    Arguments:
        :pos -- the position where the adv_picture will be saved
        :get_adv_data -- whether you need the adv_data {default false}
        :max_batch_id -- if the dataloader is big and you need to find a proper position to\
                        stop the loop {default -1}
    
    """
    #pos为攻击后图片生成的位置，之后使用的所有的pos都已预先在fastreid\data\datasets\base.py中定义了
    #使用的数据集如market duke也都预先在对应的文件定义了文件夹位置
    cfg = DefaultTrainer.auto_scale_hyperparams(cfg,data_loader.dataset.num_classes)
    model = DefaultTrainer.build_model_main(cfg)  # use baseline_train
    model.preprocess_image=change_preprocess_image(cfg) # rerange the input size

    Checkpointer(model).load(model_path)  # load trained model
    assert cfg.MODEL.ATTACKMETHOD in Attack_algorithm_library,"you need to use attack algorithm in the library, or check your spelling"

    model.eval()                #不启用batchnorm和dropout
                                #https://blog.csdn.net/qq_38410428/article/details/101102075
    adversary=match_attack_method(cfg,model,data_loader)

    for id,data in enumerate(data_loader):
        if max_batch_id!=-1 and id>=max_batch_id:
            break
        images, true_label, path = data['images'].to(device),data['targets'].to(device),data['img_paths']
                                                                            #.to("cuda")将由CPU保存的模型加载到GPU
        # data = {
        #     "images": cln_data,
        #     "targets": true_label,
        #     "camids":
        #     "img_paths": path,
        # }
        # attack
        if cfg.TARGET==False:
            with ctx_noparamgrad_and_eval(model):
                adv_data = adversary.perturb(images/255.0)  
            save_image(adv_data,path,pos) # 直接输入0-1范围的图像即可
        else :
            
            with ctx_noparamgrad_and_eval(model):
                adv_data = adversary.perturb(images/255.0,true_label)
            save_image(adv_data,path,pos) # 直接输入0-1范围的图像即可
    print('successfully generate the attack pictures of the query_set in',pos)
    


def attack_R(cfg, query_data_loader,gallery_data_loader,pos):
    
    
    model = DefaultTrainer.build_model_main(cfg)
    model.preprocess_image = change_preprocess_image(cfg) 
    model.heads = build_feature_heads(cfg)
    model.to(device)
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
    _LEGAL_ATTAKS_ = ('ES', 'QA', 'CA', 'SPQA', 'GTM', 'GTT', 'TMA', 'LTM')

    if cfg.MODEL.ATTACKMETHOD=='SMA':
        adversary=match_attack_method(cfg,model,query_data_loader)
        SMA(query_data_loader,adversary, model,pos)
    elif cfg.MODEL.ATTACKMETHOD=='FNA':
        adversary=match_attack_method(cfg,model,query_data_loader)
        FNA(query_data_loader,adversary, model,cfg.RAND,pos)
    elif cfg.MODEL.ATTACKMETHOD=='MIS-RANKING':
        mis_ranking(cfg,query_data_loader,pos)
    elif cfg.MODEL.ATTACKMETHOD=='MUAP':
        # pay attention that , if you use this attack method
        # remember to add "TEST.IMS_PER_BATCH 32" at the end of the command line to change the 
        # batch_size of query_set to 32 instead of 64
        # or you may meet CUDA out of memory error!!!
        MUAP(cfg,query_data_loader,model,pos)
    elif cfg.MODEL.ATTACKMETHOD=='SSAE':
        SSAE_attack(cfg,query_data_loader,pos)

    elif cfg.MODEL.ATTACKMETHOD in _LEGAL_ATTAKS_ or cfg.MODEL.ATTACKMETHOD[:-1] in _LEGAL_ATTAKS_:
        ROBRANK(cfg,query_data_loader,gallery_data_loader)

    else :
        raise






def match_attack_method(cfg,model,data_loader):
    mse = torch.nn.MSELoss(reduction='sum')
    def max_min_mse(f1s, f2s):
        m = 0
        for f1, f2 in zip(f1s, f2s):
            for i in range(len(f2)-1):
                m += mse(f1,f2[i])
            m -= mse(f1,f2[-1])
        return m
    
    atk_method=cfg.MODEL.ATTACKMETHOD
    #分类问题攻击算法
    if atk_method=='FGSM':
        return FGSM(model,eps = 0.05,targeted=False)
    elif atk_method =='IFGSM':
        return MomentumIterativeAttack(model, eps=0.05, eps_iter=1.0/255.0, targeted=False, decay_factor=0)
    elif atk_method =='MIFGSM':
        return MomentumIterativeAttack(model, eps=0.05, eps_iter=1.0/255.0, targeted=False, decay_factor=1.)
    elif atk_method =='CW':
        return CarliniWagnerL2Attack(model,data_loader.dataset.num_classes)
    elif atk_method =='ODFA':
        return LinfPGDAttack(model, eps=0.05, eps_iter=1.0/255.0, targeted=False, rand_init=False)

    #排序问题攻击算法
    elif atk_method=='SMA':
        return LinfPGDAttack(model,loss_fn=mse,eps=0.05,eps_iter=1.0/255.0,targeted=False, rand_init=True)
    elif atk_method=='FNA':
        return LinfPGDAttack(model,loss_fn=max_min_mse,eps=0.05, eps_iter=1.0/255.0,targeted=False, rand_init=False)
    else :
        raise KeyError("there is no attack_method you want")




