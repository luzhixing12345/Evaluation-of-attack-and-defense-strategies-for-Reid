
from math import log
from fastreid.utils.advertorch.attacks import decoupled_direction_norm
from os import path
from fastreid.utils.reid_patch import get_train_set
from fastreid.utils.attack_patch import match_attack_method
from fastreid.utils.attack_patch import attack
import torch.nn as nn
from torch import optim
import torch

from advertorch.defenses import MedianSmoothing2D
from advertorch.defenses import BitSqueezing
from advertorch.defenses import JPEGFilter
from advertorch.context import ctx_noparamgrad_and_eval
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.InputGradReg import InputGradRegLoss 

device = 'cuda'
Defense_algorithm_library=["SMOOTH2D",'ADV_DEF','GRA_REG']


def defense(cfg,query_set):
    assert cfg.MODEL.DEFENSEMETHOD in Defense_algorithm_library,"you need to use defend algorithm in the library, or check your spelling"
    if cfg.MODEL.DEFENSEMETHOD=='SMOOTH2D':
        pass
        #smooth(cfg,model,query_set)
        #DefaultTrainer.def_advtest(cfg, model) 
    elif cfg.MODEL.DEFENSEMETHOD=='ADV_DEF':
        adversarial_defense(cfg)
        query_cfg = DefaultTrainer.auto_scale_hyperparams(cfg,query_set.dataset.num_classes)
        adv_def_model = DefaultTrainer.build_model(query_cfg)  #启用baseline
        Checkpointer(adv_def_model).load('model/adv_trained.pth')  # load adv_trained model
        DefaultTrainer.test(query_cfg, adv_def_model)#用于测试对抗性训练模型的效果，并会在excel中记录，会在.out文件中输出结果
        attack(query_cfg,adv_def_model,query_set,pos='def_adv_query')
        def_adv_result =DefaultTrainer.def_advtest(query_cfg,adv_def_model)
        return def_adv_result
    elif cfg.MODEL.DEFENSEMETHOD=='GRA_REG':
        gra_reg_model=gradient_regulation(cfg)
        query_cfg = DefaultTrainer.auto_scale_hyperparams(cfg,query_set.dataset.num_classes)
        adv_def_model = DefaultTrainer.build_model(query_cfg)  #启用baseline
        Checkpointer(adv_def_model).load('model/gra_trained.pth')  # load adv_trained model
        DefaultTrainer.test(query_cfg, adv_def_model)#用于测试对抗性训练模型的效果，并会在excel中记录，会在.out文件中输出结果
        attack(query_cfg,adv_def_model,query_set,pos='def_adv_query')
        def_adv_result =DefaultTrainer.def_advtest(query_cfg,adv_def_model)
        
    # else :
    #     print('???')

# def smooth(cfg,model,query_set):
#     defense_set = nn.Sequential(
#                                 #BitSqueezing(bit_depth=5),
#                                 MedianSmoothing2D(kernel_size=3),
#                                 #JPEGFilter(10)
#                                 )
#     adversary = match_attack_method(cfg,model,query_set)
#     for data in query_set:
#         unresized_data, true_label, path = data['images'],data['targets'].to(device),data['img_paths']
#         clip_data = _resize1(unresized_data)          
#         adv_data = adversary.perturb(clip_data,true_label)
#         adv_defended=defense_set(adv_data)
#         save_image(adv_defended,path,'def_adv_query')
    

def adversarial_defense(cfg):
    
    train_data_loader = DefaultTrainer.build_train_loader(cfg) #读取训练集
    train_cfg =DefaultTrainer.auto_scale_hyperparams(cfg,train_data_loader.dataset.num_classes) 
    train_cfg.defrost()
    train_cfg.MODEL.BACKBONE.PRETRAIN = False
    model =DefaultTrainer.build_model(train_cfg)
    Checkpointer(model).load(train_cfg.MODEL.WEIGHTS)
    #optimizer = optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=1e-5)
    # optimizer = optim.SGD(model.parameters(),lr=0.001,weight_decay=1e-3) # for bot_r50
    optimizer = DefaultTrainer.build_optimizer(train_cfg, model)
    adversary = match_attack_method(train_cfg,model,train_data_loader)#对抗性攻击，找到所对抗的攻击算法

    alpha = 0.5   #混合比例
    # loss_dict = []
    # acc_clean_dict = []
    # acc_adv_dict=[]
    #total_loss = 0 
    #loss_fun = nn.CrossEntropyLoss()
    for batch_idx,data in enumerate(train_data_loader):
        if batch_idx>6000:
            break
        clean_data = data['images'].to(device)
        targets = data['targets'].to(device)
        model.eval()
        with ctx_noparamgrad_and_eval(model):
            adv_data = adversary.perturb(clean_data,targets)
        model.train()
        data['images']=(alpha*data['images']+(1-alpha)*adv_data).to(device)
        lossdict = model(data)
        loss = sum(lossdict.values())
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        # total_loss += loss.item()
        #print('batch_idx:{}/4000 Loss:{}'.format(batch_idx,loss.item()))
        # if batch_idx%200 == 0 and batch_idx!=0:
        #     acc_clean = eval_correct(model,train_data_loader)
        #     acc_adv = eval_attack_correct(model,train_data_loader,cfg)
        #     loss_dict.append(total_loss)
        #     acc_clean_dict.append(acc_clean)
        #     acc_adv_dict.append(acc_adv)
        #     print('total_loss:{} acc_clean:{} acc_adv:{}'.format(total_loss,acc_clean,acc_adv)) #每200个周期输出一次
        #     writer_idx += 1
        #     total_loss = 0
    # print('loss_dict = ',loss_dict)
    # print('acc_clean_dict = ',acc_clean_dict)
    # print('acc_adv_dict = ',acc_adv_dict)

    print('finished adv_training !')
    Checkpointer(model,'model').save('adv_trained')
    print('Successfully saved the adv_trained model !')


def gradient_regulation(cfg):
    # train a robust model again with another defense machanism

    train_data_loader = get_train_set(cfg) #读取训练集
    train_cfg = DefaultTrainer.auto_scale_hyperparams(cfg,train_data_loader.dataset.num_classes)
    train_cfg.defrost()
    train_cfg.MODEL.BACKBONE.PRETRAIN = False
    model = DefaultTrainer.build_model(train_cfg)  #启用baseline_for_defense
    Checkpointer(model).load(train_cfg.MODEL.WEIGHTS)  # load trained model
    
    #optimizer = optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=1e-5)
    #optimizer = optim.SGD(model.parameters(),lr=0.00001,weight_decay=1e-5) # for bot_r50
    optimizer = DefaultTrainer.build_optimizer(train_cfg, model)
    
    loss_fun = nn.CrossEntropyLoss()
    loss_calcuation = InputGradRegLoss(weight = 500.0,criterion = loss_fun,norm = 'L2')

    epoch = 6000
    
    model.train()
    
    for batch_idx,data in enumerate(train_data_loader):
        if(batch_idx>epoch):
            break
        clean_data = data['images']
        targets = data['targets'].to(device)
        optimizer.zero_grad()
        clean_data = clean_data.requires_grad_().to(device)
        model.eval()
        logits = model(clean_data)
        model.train()
        loss = loss_calcuation(logits,targets,clean_data) # weight的值还要好好选择一下
        loss.backward()
        optimizer.step()
        # total_loss += loss.item()
        # print('Train Epoch:{} batch_idx:{} Loss:{}'.format(writer_idx,batch_idx,loss.item()))
        # if batch_idx%200 == 0 and batch_idx!=0:
        #     loss_dict.append(total_loss)
        #     acc_clean =eval_correct(model,train_data_loader)
        #     acc_clean_dict.append(acc_clean)
        #     print('total_loss:{} acc_clean:{}'.format(total_loss,acc_clean))
        #     writer_idx += 1
        #     total_loss = 0
    
    # print('loss_dict = ',loss_dict)
    # print('acc_clean_dict = ',acc_clean_dict)

    print('finished gra_training !')
    Checkpointer(model,'model').save('gra_trained')
    print('Successfully saved the gra_trained model !')
