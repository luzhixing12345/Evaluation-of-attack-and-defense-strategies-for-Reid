
#这里引用了github的项目advtorch的库
#如想详细了解请阅读源代码
#https://github.com/BorealisAI/advertorch
from sklearn import metrics
from fastreid.utils.reid_patch import calculation_R, get_gallery_set, process_set
from fastreid.utils.advertorch.attacks import FGSM,FGM,JSMA,CarliniWagnerL2Attack,MomentumIterativeAttack,LinfPGDAttack
import torch.nn as nn
import torch
from torchvision import transforms
from advertorch.context import ctx_noparamgrad_and_eval
from fastreid.utils.attack_patch.SMA import SMA
from fastreid.utils.attack_patch.FNA import FNA
from fastreid.utils.attack_patch.MIS_RANKING import mis_ranking


device = 'cuda'
C_Attack_algorithm_library=["FGSM","CW","JSMA","FGM",'MIFGSM','ODFA','SMA','FNA','MIS-RANKING']  #针对分类问题的攻击算法库
#DDNL2 SPA时间太长了！！！


R_Attack_algorithm_library=[]
Attack_algorithm_library=C_Attack_algorithm_library+R_Attack_algorithm_library

def attack_C(cfg,model,query_set,pos='adv_query'):
    assert cfg.MODEL.ATTACKMETHOD in Attack_algorithm_library,"you need to use attack algorithm in the library, or check your spelling"

    model.eval()                #不启用batchnorm和dropout
                                #https://blog.csdn.net/qq_38410428/article/details/101102075
    adversary=match_attack_method(cfg,model,query_set)
    
    for data in query_set:
        images, true_label, path = data['images'].to(device),data['targets'].to(device),data['img_paths']
                                                                            #.to("cuda")将由CPU保存的模型加载到GPU
        # data = {
        #     "images": cln_data,
        #     "targets": true_label,
        #     "camid":
        #     "img_paths": path,
        # }
                                                    #GradientAttack中定义了
                                                    #clip_min=0., clip_max=1.
                                                    #图像需要处理数据范围
        # attack
        if cfg.TARGET==False:
            adv_untargeted_set = adversary.perturb(images)
            save_image(adv_untargeted_set/255,path,pos) # 直接输入0-1范围的图像即可
        else :
            target = torch.ones_like(true_label)      #有目标攻击直接随机指定一个target就可
            adversary.targeted = True
            with ctx_noparamgrad_and_eval(model):
                adv_targeted_set = adversary.perturb(images, target)
            save_image(adv_targeted_set/255,path,pos) # 直接输入0-1范围的图像即可


def attack_R(cfg,model,query_set):
    gallery_set=get_gallery_set(cfg)
    adversary=match_attack_method(cfg,model,query_set)
    if cfg.MODEL.ATTACKMETHOD=='SMA':
        query_features, query_pids, query_camids = process_set(query_set, model)
        gallery_features, gallery_pids, gallery_camids =SMA(gallery_set,adversary,model)  
    elif cfg.MODEL.ATTACKMETHOD=='FNA':
        gallery_features, gallery_pids, gallery_camids = process_set(gallery_set, model)
        query_features, query_pids, query_camids = FNA(query_set, adversary, model,cfg.RAND)
    elif cfg.MODEL.ATTACKMETHOD=='MIS-RANKING':
        mis_ranking(cfg,query_set,gallery_set)
    else :
        print('the attack method can not be used to attack Rank,check your spelling or R/C choosen')
        raise
    return calculation_R(cfg,query_features, query_pids, query_camids,gallery_features, gallery_pids, gallery_camids)

def save_image(imgs,paths,str):    
    imgs= imgs.cpu()                                                                 
    toPIL = transforms.ToPILImage() #这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    
    for img,path in zip(imgs,paths):  #zip 并行遍历
        pic = toPIL(img)
        position = path.find('query/') # path给的是绝对位置
        name = path[position+6:] #只需要提取出其名字即可
        pic.save(path[:position]+str+'/'+name)
    print(f"successful save in {path[:position]+str}")




def match_attack_method(cfg,model,query_set):
    mse = torch.nn.MSELoss(reduction='sum')
    def odfa(f1, f2):
        return mse(f1, -f2)
    def max_min_mse(f1s, f2s):
        # f1, f2 = fs[:,0,:], fs[:,1,:]
        # return mse(x,f1) - mse(x,f2)
        m = 0
        for f1, f2 in zip(f1s, f2s):
            for i in range(len(f2)-1):
                m += mse(f1,f2[i])
            m -= mse(f1,f2[-1])
        return m
    
    atk_method=cfg.MODEL.ATTACKMETHOD
    #分类问题攻击算法
    if atk_method=='FGSM':
        return FGSM(model,clip_max=255.0,eps = 0.05,targeted=False)
    elif atk_method=='CW':
        return CarliniWagnerL2Attack(model,num_classes=query_set.dataset.num_classes,learning_rate=0.01)
    elif atk_method=='JSMA':
        return JSMA(model,query_set.dataset.num_classes,clip_min=0.0, clip_max=255.0,loss_fn=nn.CrossEntropyLoss(reduction="sum"),gamma= 1.0)
    elif atk_method =='FGM':
        return FGM(model,loss_fn=nn.CrossEntropyLoss(reduction="sum"),eps = 0.05)
    elif atk_method =='MIFGSM':
        return MomentumIterativeAttack(model, loss_fn=torch.nn.MSELoss(reduction='sum'), eps=0.05, eps_iter=1.0/255.0, targeted=False, decay_factor=1.)
    elif atk_method =='ODFA':
        return LinfPGDAttack(model, odfa, eps=0.05, eps_iter=1.0/255.0, targeted=False, rand_init=False)

    #排序问题攻击算法
    elif atk_method=='SMA':
        return LinfPGDAttack(model,loss_fn=mse, eps=0.05, eps_iter=1.0/255.0,clip_max=255.0, targeted=False, rand_init=True)
    elif atk_method=='FNA':
        return LinfPGDAttack(model,loss_fn=max_min_mse,eps=0.05, eps_iter=1.0/255.0,clip_max=255.0, targeted=False, rand_init=False)
    else :
        print(" there is no attack_method you want ")
        raise




