
import torch
import torch.nn as nn
from torchvision import transforms
import openpyxl
from openpyxl.utils import get_column_letter
import os


from fastreid.utils.attack_patch import match_attack_method
from fastreid.data.build import build_reid_query_data_loader,build_reid_train_loader,build_reid_def_query_data_loader

import advertorch
from advertorch.attacks import GradientSignAttack,JacobianSaliencyMapAttack,L2PGDAttack,CarliniWagnerL2Attack,DDNL2Attack
from advertorch.attacks import LBFGSAttack,ElasticNetL1Attack,SinglePixelAttack,SpatialTransformAttack


device= 'cuda'
excel_name = 'result.xlsx'

def get_query_set(cfg):
    query_set=build_reid_query_data_loader(cfg,cfg.DATASETS.TESTS[0])
    return query_set

def get_def_query_set(cfg):
    def_query_set=build_reid_def_query_data_loader(cfg,cfg.DATASETS.TESTS[0])
    return def_query_set

def get_train_set(cfg):
    train_data_loader = build_reid_train_loader(cfg)
    return train_data_loader


# def check_savepath(cfg):
#     '''
#     检查生成adv_query_set的路径是否存在
#     如果已存在则删除该文件夹创建新文件夹
#     否则创建新文件夹
#     '''
#     query_path=DATASET_REGISTRY.get(cfg.DATASETS.NAMES[0]).query_dir
#     adv_query_path=os.path.abspath(os.path.join(query_path, ".."))
#     if os.path.exists(adv_query_path+"./adv_query"):
#         shutil.rmtree(adv_query_path)
#     os.path.mkdir(adv_query_path+"./adv_query")
#     return adv_query_path+"./adv_query"

def save_image(imgs,paths,str):                                                                     
    toPIL = transforms.ToPILImage() #这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    
    for img,path in zip(imgs,paths):  #zip 并行遍历
        pic = toPIL(img)
        position = path.find('query/') # path给的是绝对位置
        name = path[position+6:] #只需要提取出其名字即可
        pic.save(path[:position]+str+'/'+name)
    print(f"successful save in {path[:position]+str}")

def _resize1(unresized_data):
    return unresized_data/255.0



# @torch.no_grad()
# def eval_correct(model,train_data_loader):
#     model.eval()
#     correct = 0 
#     softmax = nn.Softmax(dim=1)
#     for batch_idx,data in enumerate(train_data_loader):
#         if(batch_idx>=200):
#             break
#         outputs = model(data['images']) # attack时输入的范围时0-1
#         probabilities = softmax(outputs)
#         pred = probabilities.argmax(dim=1,keepdim=True)
#         targets = data['targets'].to(device)
#         correct += pred.eq(targets.view_as(pred)).sum().item()
#     return 100. * correct /(64*200)

# @torch.no_grad()
# def eval_attack_correct(model,train_data_loader,cfg):
#     model.eval()
#     correct = 0 
#     softmax = nn.Softmax(dim=1)
#     adversary = match_attack_method(cfg,model,train_data_loader)
#     for batch_idx,data in enumerate(train_data_loader):
#         if(batch_idx>=200):
#             break
#         targets = data['targets'].to(device)
#         with torch.enable_grad():
#             data = adversary.perturb(_resize1(data['images']),targets)
#         logits = model(data)
#         probabilities = softmax(logits)
#         pred = probabilities.argmax(dim=1,keepdim=True)
#         correct += pred.eq(targets.view_as(pred)).sum().item()
#     return 100. * correct / (64*200)

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
    for row in range (3,4+len(Attack_algorithm_library)*6):
        for col in range (2,6+len(Defense_algorithm_library)):
            sheet.column_dimensions[get_column_letter(col)].width = 20.0
            sheet.row_dimensions[row].height = 40
    sheet['D3']='PURE_VALUE'
    sheet['E3']='AFTER_ATTACK'
    for row,name in enumerate(Attack_algorithm_library):
        sheet['B'+str(4+6*row)]=name
        sheet['C'+str(4+6*row)]='Rank-1'
        sheet['C'+str(5+6*row)]='Rank-5'
        sheet['C'+str(6+6*row)]='Rank-10'
        sheet['C'+str(7+6*row)]='mAP'
        sheet['C'+str(8+6*row)]='mINP'
        sheet['C'+str(9+6*row)]='metric'

    for col,name in enumerate(Defense_algorithm_library,start=6):
        sheet[get_column_letter(col)+str(3)]=name

def save_data(cfg,pure_result,adv_result,def_result,sheet):
    from fastreid.utils.attack_patch import Attack_algorithm_library
    from fastreid.utils.defense_patch import Defense_algorithm_library

    row_start = 4 + 6*Attack_algorithm_library.index(cfg.MODEL.ATTACKMETHOD)
    column = chr(ord('F') + Defense_algorithm_library.index(cfg.MODEL.DEFENSEMETHOD))

    for col,dict_name in {'4':pure_result,'5':adv_result,column:def_result}.items():
        for row in range(row_start,row_start+6):
            sheet[col+str(row)]=dict_name[sheet['C'+str(row)]]

# def save_config(cfg,pure_result,adv_result,def_result,sheet):

#     start_row = str(sheet.max_row+2)
#     sheet['B'+start_row]='PURE'
#     sheet['C'+start_row]=cfg.attack_method
#     sheet['D'+start_row]=cfg.defense_method

#     start_row = str(int(start_row)+1)
#     sheet['B'+start_row]=pure_result
#     sheet['C'+start_row]=adv_result
#     sheet['D'+start_row]=def_result

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

def record(cfg,pure_result,adv_result,def_result,save_pos= excel_name):
    if check(save_pos):
        wb = openpyxl.load_workbook(save_pos)
    else :
        wb = openpyxl.Workbook()
    sheet_list = wb.sheetnames
    sheet_name = cfg.DATASETS.NAMES[0] + '_'+cfg.MODEL.BACKBONE.NAME 
    if sheet_name in sheet_list:
        sheet = wb[sheet_name]
    else :
        sheet = wb.create_sheet(title = sheet_name)
        sheet_init(sheet)

    save_data(cfg,pure_result,adv_result,def_result,sheet)
    #save_config(cfg,pure_result,adv_result,def_result,sheet)
    remove_Sheet(wb,sheet_list)
    wb.save(save_pos)