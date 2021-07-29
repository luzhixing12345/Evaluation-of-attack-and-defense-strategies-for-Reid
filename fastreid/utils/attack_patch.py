
#这里引用了github的项目advtorch的库
#如想详细了解请阅读源代码
#https://github.com/BorealisAI/advertorch
from fastreid.utils.advertorch.attacks import decoupled_direction_norm
from fastreid.utils import advertorch
from .advertorch.utils import predict_from_logits
from .advertorch.attacks import GradientSignAttack,JacobianSaliencyMapAttack,CarliniWagnerL2Attack,DDNL2Attack
from .advertorch.attacks import LBFGSAttack,ElasticNetL1Attack,SinglePixelAttack,SpatialTransformAttack
import torch.nn as nn
import torch
from torchvision import transforms
from advertorch.context import ctx_noparamgrad_and_eval
from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer

device = 'cuda'
Attack_algorithm_library=["FGSM","L2PGD","CW","JSMA","DDNL2","ENL1","SP","ST","LBFGS"]

def attack(cfg,model,query_set,pos='adv_query'):
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
            adv_untargeted_set = adversary.perturb(images, true_label)
            save_image(adv_untargeted_set/255,path,pos) # 直接输入0-1范围的图像即可
        else :
            target = torch.ones_like(true_label) * 3      #有目标攻击直接随机指定一个target就可
            adversary.targeted = True
            with ctx_noparamgrad_and_eval(model):
                adv_targeted_set = adversary.perturb(images, target.to(device))
            save_image(adv_targeted_set,path,pos) # 直接输入0-1范围的图像即可


        

def save_image(imgs,paths,str):    
    imgs= imgs.cpu()                                                                 
    toPIL = transforms.ToPILImage() #这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    
    for img,path in zip(imgs,paths):  #zip 并行遍历
        pic = toPIL(img)
        position = path.find('query/') # path给的是绝对位置
        name = path[position+6:] #只需要提取出其名字即可
        pic.save(path[:position]+str+'/'+name)
    print(f"successful save in {path[:position]+str}")

def _resize1(unresized_data):
    return unresized_data/255.0

def match_attack_method(cfg,model,query_set):
    atk_method=cfg.MODEL.ATTACKMETHOD
    if atk_method=='FGSM':
        return GradientSignAttack(model,loss_fn=nn.CrossEntropyLoss(reduction="sum"),eps = 0.05)
    elif atk_method=='CW':
        return CarliniWagnerL2Attack(model,num_classes=query_set.dataset.num_classes,loss_fn=nn.CrossEntropyLoss(reduction="sum"),learning_rate=0.01)
    elif atk_method=='JSMA':
        return JacobianSaliencyMapAttack(model,query_set.dataset.num_classes,clip_min=0.0, clip_max=1.0,loss_fn=nn.CrossEntropyLoss(reduction="sum"),gamma= 1.0)
    elif atk_method=='DDNL2':
        return DDNL2Attack(model,loss_fn=nn.CrossEntropyLoss(reduction="sum"))
    elif atk_method=='ENL1':
        return ElasticNetL1Attack(model,loss_fn=nn.CrossEntropyLoss(reduction="sum"),learning_rate=1e-2)
    elif atk_method=='SP':
        return SinglePixelAttack(model,loss_fn=nn.CrossEntropyLoss(reduction="sum"))
    elif atk_method=='ST':
        return SpatialTransformAttack(model,query_set,search_steps=1,loss_fn=nn.CrossEntropyLoss(reduction="sum"))
    elif atk_method=='LBFGS':
        return LBFGSAttack(model,num_classes=query_set.dataset.num_classes,loss_fn=nn.CrossEntropyLoss(reduction="sum"))
    else :
        print(" there is no attack_method you want ")
        return


