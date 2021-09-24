
import os
from sys import path
import time

import openpyxl
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from fastreid.data.build import (build_reid_def_query_data_loader,
                                 build_reid_query_data_loader,
                                 build_reid_gallery_data_loader,
                                 build_reid_train_loader)
from fastreid.engine import DefaultTrainer
from fastreid.utils.compute_dist import build_dist
from fastreid.evaluation.rank import evaluate_rank
from fastreid.utils.checkpoint import Checkpointer
from openpyxl.utils import get_column_letter
from torchvision import transforms


device= 'cuda'
excel_name = 'result.xlsx'
txt_name = 'result.txt'

def get_query_set(cfg):
    query_set=build_reid_query_data_loader(cfg,cfg.DATASETS.TESTS[0])
    return query_set

def get_def_query_set(cfg):
    def_query_set=build_reid_def_query_data_loader(cfg,cfg.DATASETS.TESTS[0])
    return def_query_set

def get_gallery_set(cfg):
    gallery_set=build_reid_gallery_data_loader(cfg,cfg.DATASETS.TESTS[0])
    return gallery_set
def get_train_set(cfg):
    train_data_loader = build_reid_train_loader(cfg)
    return train_data_loader

def get_pure_result(cfg,query_set):
    cfg = DefaultTrainer.auto_scale_hyperparams(cfg,query_set.dataset.num_classes)
    std_model = DefaultTrainer.build_model(cfg)
    Checkpointer(std_model).load(cfg.MODEL.WEIGHTS)
    pure_result=DefaultTrainer.test(cfg, std_model)       #test用于测试原始的query与gallery合成的test_set
    return pure_result,std_model

@torch.no_grad()
def eval_train(model,query_data_loader,max_batch_id=-1):
    model.eval()
    correct = 0 
    softmax = nn.Softmax(dim=1)
    for batch_idx,data in enumerate(query_data_loader):
        if max_batch_id!=-1 and batch_idx>=max_batch_id:
            break
        logits = model(data)
        probabilities = softmax(logits)
        pred = probabilities.argmax(dim=1,keepdim=True)
        targets = data['targets'].to(device)
        correct += pred.eq(targets.view_as(pred)).sum().item()

    if max_batch_id==-1:
        l=len(query_data_loader.dataset)
    else :
        l=max_batch_id*64
    return 100. * correct / l

def evaluate_misMatch(model,query_data_loader):
    return 0
    # correct = 0 
    # softmax = nn.Softmax(dim=1)
    # for batch_idx,data in enumerate(query_data_loader):
    #     logits = model(data)
    #     probabilities = softmax(logits)
    #     pred = probabilities.argmax(dim=1,keepdim=True).to("cpu")
    #     targets = torch.ones_like(data['targets'].to("cpu")).to("cpu")
    #     correct += pred.eq(targets.view_as(pred)).sum().item()

    # return 100. * correct / len(query_data_loader.dataset)


def save_image(imgs,paths,str):                                                                     
    toPIL = transforms.ToPILImage() #这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    
    for img,path in zip(imgs,paths):  #zip 并行遍历
        pic = toPIL(img)
        position = path.find('query/') # path给的是绝对位置
        name = path[position+6:] #只需要提取出其名字即可
        pic.save(path[:position]+str+'/'+name)
    print(f"successful save in {path[:position]+str}")

def train_query_set(cfg,query_data_loader):
    '''
    固定住backbone的权重，利用query_set重新训练cls_layer的weight,使其也能准确的分类query_set 中的图像和id
    转换成分类问题，可较好的接收针对于分类问题的算法攻击
    
    '''
    query_cfg = DefaultTrainer.auto_scale_hyperparams(cfg,query_data_loader.dataset.num_classes)
    model = DefaultTrainer.build_model_for_pretrain(query_cfg)  #启用baseline_for_train_query
    Checkpointer(model).load(query_cfg.MODEL.WEIGHTS)  # load trained model

    optimizer = optim.Adam(filter(lambda p: p.requires_grad ,model.parameters()),lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=1e-5) 
    loss_fun = nn.CrossEntropyLoss()

    for _,parm in enumerate(model.parameters()):#固定除分类层其他层的权重，在训练中不计算梯度
        parm.requires_grad=False
    for _,parm in enumerate(model.heads.classifier.parameters()):
        parm.requires_grad=True
    #分类层参数清空（正态分布）
    torch.nn.init.xavier_normal_(model.heads.classifier.weight)
    loss_fun = nn.CrossEntropyLoss()

    epoch = 5
    for i in range(epoch):
        model.train()
        for batch_idx,data in enumerate(query_data_loader):
            optimizer.zero_grad()
            targets = data['targets'].to(device)
            logits = model(data)
            loss = loss_fun(logits,targets)
            loss.backward()
            optimizer.step()
            
        accurency = eval_train(model,query_data_loader)
        print('The accurency for Train Epoch {} is {}'.format(i,accurency))
        print('---------------------------------------------------------------------')
    Checkpointer(model,'model').save('query_trained')
    return cfg

def train_train_set(cfg,train_data_loader):
    '''
    固定住backbone的权重，利用train_set重新训练cls_layer的weight ,使其也能准确的分类train_set 中的图像和id
    转换成分类问题，可较好的接收针对于分类问题的算法攻击
    
    '''
    train_cfg = DefaultTrainer.auto_scale_hyperparams(cfg,train_data_loader.dataset.num_classes)
    model = DefaultTrainer.build_model_for_pretrain(train_cfg)  #启用baseline_for_train_query
    Checkpointer(model).load(train_cfg.MODEL.QUERYSET_TRAINED_WEIGHT)  # load trained model
    optimizer = DefaultTrainer.build_optimizer(train_cfg, model)
    for idx,parm in enumerate(model.parameters()):#固定除分类层其他层的权重，在训练中不计算梯度
        parm.requires_grad=False
    for idx,parm in enumerate(model.heads.classifier.parameters()):
        parm.requires_grad=True
    #分类层参数清空（正态分布）
    #torch.nn.init.xavier_normal_(model.heads.cls_layer.weight)
    loss_fun = nn.CrossEntropyLoss()

    model.train()
    for batch_idx,data in enumerate(train_data_loader):
        if batch_idx>=4000:
            break
        optimizer.zero_grad()
        targets = data['targets'].to(device)
        logits = model(data)
        loss = loss_fun(logits,targets)
        loss.backward()
        optimizer.step()
        if batch_idx%1000==0:
            accurency = eval_train(model,train_data_loader,400)
            print('-------------------------------------')
            print(f'The accurency for the training set is {accurency}')
            print('-------------------------------------')

    Checkpointer(model,'model').save('pretrained')
    print('You finish the pretrain for the training set, and the model was saved in ./model/pretrain_model.pth')
    print('-----------------------------------')




def check(save_pos):
    return os.path.exists(save_pos)

def remove_Sheet(wb,sheet_list):
    if 'Sheet' in sheet_list:
        wb.remove(wb['Sheet'])
    if 'Sheet1' in sheet_list:
        wb.remove(wb['Sheet1'])
def sheet_init(sheet):
    from fastreid.utils.attack_patch import Attack_algorithm_library
    from fastreid.utils.defense_patch import Defense_algorithm_library
    for row in range (3,4+len(Attack_algorithm_library)*7):
        for col in range (2,6+len(Defense_algorithm_library)*2):
            sheet.column_dimensions[get_column_letter(col)].width = 20.0
            sheet.row_dimensions[row].height = 40
    sheet['D3']='PURE_VALUE'
    sheet['E3']='AFTER_ATTACK'
    for row,name in enumerate(Attack_algorithm_library):
        sheet['B'+str(4+7*row)]=name
        sheet['C'+str(4+7*row)]='Rank-1'
        sheet['C'+str(5+7*row)]='Rank-5'
        sheet['C'+str(6+7*row)]='Rank-10'
        sheet['C'+str(7+7*row)]='mAP'
        sheet['C'+str(8+7*row)]='mINP'
        sheet['C'+str(9+7*row)]='metric'
        sheet['C'+str(10+7*row)]='misMatch'

    for col,name in enumerate(Defense_algorithm_library):
        sheet[get_column_letter(6+2*col)+str(3)]='AFTER_DEFENSE'
        sheet[get_column_letter(7+2*col)+str(3)]=name

def save_data(cfg,pure_result,adv_result,def_result,def_adv_result,sheet):
    from fastreid.utils.attack_patch import Attack_algorithm_library
    from fastreid.utils.defense_patch import Defense_algorithm_library

    row_start = 4 + 7*Attack_algorithm_library.index(cfg.MODEL.ATTACKMETHOD)
    column1 = chr(ord('F') + Defense_algorithm_library.index(cfg.MODEL.DEFENSEMETHOD)*2)
    column2 = chr(ord(column1)+1)

    for col,dict_name in {'D':pure_result,'E':adv_result,column1:def_result,column2:def_adv_result}.items():
        sheet[col+str(row_start)]  =dict_name['Rank-1']
        sheet[col+str(row_start+1)]=dict_name['Rank-5']
        sheet[col+str(row_start+2)]=dict_name['Rank-10']
        sheet[col+str(row_start+3)]=dict_name['mAP']
        sheet[col+str(row_start+4)]=dict_name['mINP']
        sheet[col+str(row_start+5)]=dict_name['metric']
        sheet[col+str(row_start+6)]=dict_name['misMatch']

# def save_config(cfg,pure_result,adv_result,def_adv_result,sheet):

#     start_row = str(sheet.max_row+2)
#     sheet['B'+start_row]='PURE'
#     sheet['C'+start_row]=cfg.attack_method
#     sheet['D'+start_row]=cfg.defense_method

#     start_row = str(int(start_row)+1)
#     sheet['B'+start_row]=pure_result
#     sheet['C'+start_row]=adv_result
#     sheet['D'+start_row]=def_adv_result

#     start_row =int(start_row)+2
#     sheet['B'+str(start_row)]='ATTACK_ARGUMENTS'
#     for col in range (len(cfg.attack_arguments)):
#         sheet.cell(column= 3+2*col,row=start_row,value=list(cfg.attack_arguments.keys())[col])
#         sheet.cell(column= 4+2*col,row=start_row,value=list(cfg.attack_arguments.values())[col])

#     start_row =int(start_row)+2
#     sheet['B'+str(start_row)]='DEFENSE_ARGUMENTS'
#     for col in range (len(cfg.defense_arguments)):
#         sheet.cell(column= 3+2*col,row=start_row,value=list(cfg.defense_arguments.keys())[col])
#         sheet.cell(column= 4+2*col,row=start_row,value=list(cfg.defense_arguments.values())[col])

def record(cfg,pure_result,adv_result,def_result,def_adv_result,save_pos= excel_name):
    if check(save_pos):
        wb = openpyxl.load_workbook(save_pos)
    else :
        wb = openpyxl.Workbook()
    sheet_list = wb.sheetnames
    sheet_name = cfg.DATASETS.NAMES[0]
    if cfg.TARGET :
        sheet_name+='_target'
    else :
        sheet_name+='_untarget'
    if sheet_name in sheet_list:
        sheet = wb[sheet_name]
    else :
        sheet = wb.create_sheet(title = sheet_name)
        sheet_init(sheet)

    save_data(cfg,pure_result,adv_result,def_result,def_adv_result,sheet)
    #save_config(cfg,pure_result,adv_result,def_adv_result,sheet)
    remove_Sheet(wb,sheet_list)
    wb.save(save_pos)

def process_set(loader, model, device='cuda'):
    """Extract all the features of images from a loader using a model.
    
    Arguments:
        loader {Pytorch dataloader} -- loader of the images
        model {Pytorch model} -- model used to extract the features
        device {cuda device} -- 
    
    Returns:
        features -- Tensor of the features of the queries
        ids -- numpy array of ids
        cams -- numpy array of camera ids
    """

    ids = []
    cams = []
    features = []
    model.eval()
    for _, data in enumerate(loader):
        with torch.no_grad():
            output = model(data['image'].to(device))
        features.append(output)
        ids.append(data['targets'].cpu())
        cams.append(data['camid'].cpu())
    ids = torch.cat(ids, 0)
    cams = torch.cat(cams, 0)
    features = torch.cat(features, 0)
    return features.cpu(), ids.numpy(), cams.numpy()




def calculation_R(cfg,query_features, query_pids, query_camids,gallery_features, gallery_pids, gallery_camids):
    results={}
    dist = build_dist(query_features, gallery_features, cfg.TEST.METRIC)
    cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    for r in [1, 5, 10]:
        results['Rank-{}'.format(r)] = cmc[r - 1] * 100
    results['mAP'] = mAP * 100
    results['mINP'] = mINP * 100
    results["metric"] = (mAP + cmc[0]) / 2 * 100
    print('--------------------------')
    print('RANK-1 = ',results['Rank-1'])
    print('--------------------------')
    return results

def pairwise_distance(x, y):
    """Compute the matrix of pairwise distances between tensors x and y

    Args:
        x (Tensor)
        y (Tensor)

    Returns:
        Tensor: matrix of pairwise distances
    """
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()
    return dist