
import copy
import time
import os
import shutil

import openpyxl
import torch
import torch.nn as nn
from fastreid.data.build import (build_reid_test_data_loader,
                                 build_reid_query_data_loader,
                                 build_reid_att_query_data_loader,
                                 build_reid_gallery_data_loader,
                                 build_reid_train_loader,)
from fastreid.engine import DefaultTrainer
from fastreid.solver.build import build_optimizer
from fastreid.utils.checkpoint import Checkpointer
from openpyxl.utils import get_column_letter
from torchvision import transforms
from fastreid.engine import DefaultTrainer

device= 'cuda'
excel_name = 'result.xlsx'

C_Attack_algorithm_library=["C-FGSM",'C-IFGSM','C-MIFGSM','CW']  #针对分类问题的攻击算法库
R_Attack_algorithm_library=['R-FGSM','R-IFGSM','R-MIFGSM','ODFA','MISR','FNA','MUAP','SSAE','ES','GTM', 'GTT', 'TMA', 'LTM']
Attack_algorithm_library=C_Attack_algorithm_library+R_Attack_algorithm_library

G_Defense_algorithm_library=['ADV_DEF','GRA_REG','DISTILL']
R_Defense_algorithm_library=['GOAT','EST','SES','PNP']
Defense_algorithm_library=G_Defense_algorithm_library+R_Defense_algorithm_library

evaluation_indicator=['Rank-1','Rank-5','Rank-10','mAP','mINP','metric']
evaluation_attackIndex=['mAP','SDSIM','index']

attack_type = ['QA+','QA-','GA+','GA-']
num=len(evaluation_indicator)

def get_query_set(cfg,relabel=True):
    query_set=build_reid_query_data_loader(cfg,cfg.DATASETS.TESTS[0],relabel=relabel)
    return query_set

def get_att_query_set(cfg,relabel=True):
    adv_query_set = build_reid_att_query_data_loader(cfg,cfg.DATASETS.TESTS[0],relabel=relabel)
    return adv_query_set

def get_test_set(cfg,relabel=True):
    test_set = build_reid_test_data_loader(cfg,cfg.DATASETS.TESTS[0],relabel=relabel)
    return test_set

def get_gallery_set(cfg,relabel=True):
    gallery_set=build_reid_gallery_data_loader(cfg,cfg.DATASETS.TESTS[0],relabel=relabel)
    return gallery_set

def get_train_set(cfg):
    train_data_loader = build_reid_train_loader(cfg)
    return train_data_loader



@torch.no_grad()
def eval_train(model,data_loader,max_id=-1):
    model.eval()
    correct = 0 
    softmax = nn.Softmax(dim=1)
    for id,data in enumerate(data_loader):
        if max_id!=-1 and id>max_id:
            break
        logits = model(data)
        probabilities = softmax(logits)
        pred = probabilities.argmax(dim=1,keepdim=True)
        targets = data['targets'].to(device)
        correct += pred.eq(targets.view_as(pred)).sum().item()

    if max_id==-1:
        l=len(data_loader.dataset)
    else:
        l=max_id*64
    return 100. * correct / l


def save_image(imgs,paths,str):
      
    imgs= imgs.cpu()                                                              
    toPIL = transforms.ToPILImage() #这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    
    for img,path in zip(imgs,paths):
        pic = toPIL(img)
        position = path.find('query/') # path给的是绝对位置
        name = path[position+6:] #只需要提取出其名字即可
        pic.save(path[:position]+str+'/'+name)

    print(f"successful save in {path[:position]+str}")



def classify_test_set(cfg,data_loader):
    '''
    固定住backbone的权重，利用query_set重新训练cls_layer的weight,使其也能准确的分类图像和id
    转换成分类问题，可较好的接收针对于分类问题的算法攻击
    
    '''
    #set the number of neurons of the classifier layer as query set's targets number
    cfg = DefaultTrainer.auto_scale_hyperparams(cfg,data_loader.dataset.num_classes)

    model = DefaultTrainer.build_model_main(cfg)  # use baseline_train
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

    optimizer = build_optimizer(cfg,model) 
    loss_fun = nn.CrossEntropyLoss()
    #fix the weight of the backbone layers,and we only train the classifier for query set
    for _,parm in enumerate(model.parameters()):
        parm.requires_grad=False
    for _,parm in enumerate(model.heads.classifier.parameters()):
        parm.requires_grad=True

    # the origin model was trained by training set ,which has different targets number from query set,
    # so the number of neurons of the classifier layer was different, the weight couldn'y directly be loaded,
    # and you can see the warning like 
    # "Skip loading parameter 'heads.classifier.weight' to the model due to incompatible shapes: (751, 2048) in the checkpoint but (0, 2048) in the model! You might want to double check if this is expected.'
    # so we skip this part of weight and initial weights as below
    torch.nn.init.xavier_normal_(model.heads.classifier.weight)
    loss_fun = nn.CrossEntropyLoss()

    epoch = 5  #as far as i noticed,within 5 epoches the classifier layer can be trained to efficiently behave well
    for i in range(epoch):
        model.train()
        for _,data in enumerate(data_loader):
            targets = data['targets'].to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = loss_fun(logits,targets)
            loss.backward()
            optimizer.step()
            
        accurency = eval_train(model,data_loader) #show the accurrency
        print('The accurency of query set in Train Epoch {} is {}'.format(i,accurency))
        print('---------------------------------------------------------------------')
    Checkpointer(model,'model').save('test_trained') # model was saved in ./model/test_trained.pth
    print('the model was saved in ./model/test_trained.pth')

def match_type(cfg,type):
    
    if type == 'attack':
        atk_method=cfg.ATTACKMETHOD
        if atk_method in C_Attack_algorithm_library:
            return True
        elif atk_method in R_Attack_algorithm_library:
            return False
        else:
            raise KeyError('you should use the attack method in the library, or check your spelling')
    else:
        def_method=cfg.DEFENSEMETHOD
        if def_method in G_Defense_algorithm_library:
            return True
        elif def_method in R_Defense_algorithm_library:
            return False
        else :
            raise KeyError('you should use the defense method in the library, or check your spelling')


def change_preprocess_image(cfg):
    def preprocess_image(batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(device)
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))
        
        
        images = torch.mul(images,255)
        images.sub_(torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1).to(device)).div_(torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1).to(device))
        return images
    return preprocess_image
    


def release_cuda_memory():
    if hasattr(torch.cuda, 'empty_cache'):
	    torch.cuda.empty_cache()

def get_result(cfg,model_path,step:str)->dict:
    model =DefaultTrainer.build_model(cfg)  # 启用baseline,用于测评
    Checkpointer(model).load(model_path)
    if step=='attack':
        result = DefaultTrainer.advtest(cfg,model)# advtest用于测试adv_query与gallery合成的test_set
    elif step=='pure' or step=='defense':
        result = DefaultTrainer.test(cfg,model)# test用于测试query与gallery合成的test_set
    elif step =='def-attack':
        result = DefaultTrainer.def_advtest(cfg,model)# def-advtest用于测试def-adv_query与gallery合成的test_set
    else:
        raise KeyError('you must choose the step to select the correct dataset of query and gallery for evaluation.')
    return result



def check(save_pos):
    return os.path.exists(save_pos)



def remove_Sheet(wb,sheet_list):
    if 'Sheet' in sheet_list:
        wb.remove(wb['Sheet'])
    if 'Sheet1' in sheet_list:
        wb.remove(wb['Sheet1'])
def sheet_init(sheet):
    for row in range (3,4+len(Attack_algorithm_library)*num):
        for col in range (2,6+len(Defense_algorithm_library)*2):
            sheet.column_dimensions[get_column_letter(col)].width = 20
            sheet.row_dimensions[row].height = 40
    sheet['D3']='PURE VALUE'
    sheet['E3']='AFTER ATTACK'
    for row,name in enumerate(Attack_algorithm_library):
        sheet['B'+str(4+num*row)]=name
        for i in range(4,4+num):
            sheet['C'+str(i+num*row)]=evaluation_indicator[i-4]

    for col,name in enumerate(Defense_algorithm_library):
        sheet[get_column_letter(6+2*col)+str(3)]=name
        sheet[get_column_letter(7+2*col)+str(3)]='ATFER ATTACK'

def sheet_AI_init(sheet_AI):
    for row in range (3,4+2*len(C_Attack_algorithm_library)+4*len(R_Attack_algorithm_library)):
        for col in range (2,7):
            sheet_AI.column_dimensions[get_column_letter(col)].width = 20
            sheet_AI.row_dimensions[row].height = 40

    bias = 4
    for row,name in enumerate(C_Attack_algorithm_library):
        sheet_AI['B'+str(bias+2*row)]=name
        sheet_AI['B'+str(bias+1+2*row)]=name
        sheet_AI['C'+str(bias+2*row)]='UT'
        sheet_AI['C'+str(bias+2*row+1)]='T'
    
    bias = 4+2*len(C_Attack_algorithm_library)
    for row,name in enumerate(R_Attack_algorithm_library):
        for i in range(4):
            sheet_AI['B'+str(bias+i+4*row)]=name
        for i,type in enumerate(attack_type):
            sheet_AI['C'+str(bias+4*row+i)] =type
    
    for i in range(len(evaluation_attackIndex)):
        sheet_AI[get_column_letter(i+4)+'3'] = evaluation_attackIndex[i]

def save_data(cfg,pure_result,att_result,def_result,def_adv_result,sheet):

    row_start = 4 + num*Attack_algorithm_library.index(cfg.ATTACKMETHOD)
    column1 = chr(ord('F') + Defense_algorithm_library.index(cfg.DEFENSEMETHOD)*2)
    column2 = chr(ord(column1)+1)

    for col,dict_name in {'D':pure_result,'E':att_result,column1:def_result,column2:def_adv_result}.items():
        for i in range(num):
            sheet[col+str(row_start+i)]  =dict_name[evaluation_indicator[i]]

def save_attack_index(cfg,pure_result,att_result,SSIM,sheet):
    if cfg.ATTACKMETHOD in C_Attack_algorithm_library:
        row_start = 4 + C_Attack_algorithm_library.index(cfg.ATTACKMETHOD)
        bias = 0 if cfg.ATTACKTYPE=='UT'else 1
        row = row_start+bias
    else:
        row_start = 4 + 2*len(C_Attack_algorithm_library)+4*R_Attack_algorithm_library.index(cfg.ATTACKMETHOD)
        bias = attack_type.index(cfg.ATTACKTYPE+cfg.ATTACKDIRECTION)
        row = row_start+bias
    
    column = 4
    
    ans = {}
    ans['mAP'] = pure_result['mAP']-att_result['mAP']
    ans['SDSIM'] = 1-SSIM
    ans['index'] = ans['mAP']/ans['SDSIM']

    for i,indicator in enumerate(evaluation_attackIndex):
        sheet[get_column_letter(column+i)+str(row)] =ans[indicator]


def record(cfg,pure_result,att_result,def_result,def_adv_result,SSIM,save_pos= excel_name):
    if check(save_pos):
        wb = openpyxl.load_workbook(save_pos)
    else :
        wb = openpyxl.Workbook()
    sheet_list = wb.sheetnames
    sheet_name = cfg.DATASETS.NAMES[0]
    if sheet_name in sheet_list:
        sheet = wb[sheet_name]
    else :
        sheet = wb.create_sheet(title = sheet_name)
    
    sheet_attackIndex_name='attackIndex'
    if sheet_attackIndex_name in sheet_list:
        sheet_AI = wb[sheet_attackIndex_name]
    else :
        sheet_AI = wb.create_sheet(title = sheet_attackIndex_name)
    sheet_init(sheet)
    sheet_AI_init(sheet_AI)
    if def_result!=None or def_adv_result != None:
        save_data(cfg,pure_result,att_result,def_result,def_adv_result,sheet)
    if pure_result!=None and att_result!=None and SSIM !=None:
        save_attack_index(cfg,pure_result,att_result,SSIM,sheet_AI)
    #save_config(cfg,pure_result,adv_result,def_adv_result,sheet)
    remove_Sheet(wb,sheet_list)
    wb.save(save_pos)

def process_set(loader, model,device='cuda'):
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
    for id, data in enumerate(loader):
        with torch.no_grad():
            output = model((data['images']/255.0).to(device))
        features.append(output)
        ids.append(data['targets'].cpu())
        cams.append(data['camids'].cpu())
    ids = torch.cat(ids, 0)
    cams = torch.cat(cams, 0)
    features = torch.cat(features, 0)
    return features.cpu(), ids.numpy(), cams.numpy()


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

def CHW_to_HWC(images):
    # compare_ssim function in sklearn is used for HWC images
    # this function was used to transform CHW images to HWC images
    # (size)3x256x128 -> (size)256x128x3
    images = images.cpu()
    images = images.permute(1,2,0)
    # tensor.permute()
    # change dimension of tensor, only tensor
    # previous dimension C(0),H(1),W(2) to H(1),W(2),C(0) 
    return images.detach().numpy()


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        print("dirtory "+path+" has been created")
    else:
        if os.path.exists(path+'/origin'):
            shutil.rmtree(path+"/origin")
        if os.path.exists(path+'/attack'):
            shutil.rmtree(path+"/attack")
        if os.path.exists(path+'/defense'):
            shutil.rmtree(path+"/defense")
        if os.path.exists(path+'/adv_defense'):
            shutil.rmtree(path+"/adv_defense")
    os.makedirs(path+"/origin") 
    os.makedirs(path+"/attack") 
    os.makedirs(path+"/defense")
    os.makedirs(path+"/adv_defense")


def record_order(cfg,pure_result,att_result,def_result,def_adv_result, pictureNumber = 50,save_pic=False):

    batch_size = cfg.TEST.IMS_PER_BATCH

    # query_set = get_query_set(cfg,relabel=False)
    # gallery_set = get_gallery_set(cfg,relabel=False)
    
    # gallery_images = []
    # gallery_targets = []
    # for _ ,data in enumerate(gallery_set):
    #     gallery_images.append(data['images'].cpu()/255.0)
    #     gallery_targets.append(data['targets'])

    test_dataset,num_query = DefaultTrainer.build_test_loader(cfg,dataset_name=cfg.DATASETS.NAMES[0])
    images = []
    pids = []
    camids = []
    print('batch size = ',batch_size)
    print('num_query = ',num_query)

    for _,data in enumerate(test_dataset):
        images.append((data['images']/255.0).cpu())
        pids.append(data['targets'].cpu())
        camids.append(data['camids'].cpu())
    
    images = torch.cat(images,dim=0)
    pids = torch.cat(pids, dim=0)
    camids = torch.cat(camids,dim=0)

    query_images = images[:num_query]
    query_pids = pids[:num_query]

    gallery_images = images[num_query:]
    gallery_pids = pids[num_query:]

    result_order = {}
    q_pid_save = {}
    g_pids_save = {}
    for name,dict in {'origin':pure_result,'attack':att_result,'defense':def_result,'adv_defense':def_adv_result}.items():
        result_order[name]=[]
        q_pid_save[name]=None
        g_pids_save[name]=[]
        if dict!=None:
            for key,list in {'result_order':result_order,'q_pid_save':q_pid_save,'g_pids_save':g_pids_save}.items():
                list[name]=dict[key]

    pictureNumber = pictureNumber  
    #the same as 'max_rank' in engine/evaluation/rank.py  -> function(eval_cuhk/eval_market1501)

    
    path = f"{os.getcwd()}/logs_pic/{cfg.DATASETS.NAMES[0]}/{cfg.ATTACKTYPE}{cfg.ATTACKDIRECTION}/{cfg.ATTACKMETHOD}_{cfg.DEFENSEMETHOD}"
    mkdir(path)

    log(cfg,q_pid_save,g_pids_save,pictureNumber)
    
    
    # start to save pictures

    # save method 1
    # toPIL = transforms.ToPILImage()

    # for name,pos in result_order.items():
    #     print(f'start to save pictures of query&gallery set of {name}')
    #     for _,data in enumerate(query_set):
    #         for i in range(pictureNumber):
    #             img = (data['images'].cpu()/255.0)[i]
    #             target = data['targets'][i].item()
    #             os.makedirs(f'{path}/{name}/{i}')
    #             toPIL(img).save(f'{path}/{name}/{i}/q_{target}.jpg')
    #             if q_pid_save[name]==None:
    #                 continue
    #             if target!=q_pid_save[name][i]:
    #                 print("target =",target)
    #                 print('q_pid_save = ',q_pid_save[name][i])
    #         break
    #     print('query set pictures are all saved ')
    #     if pos==[]:
    #         continue
    #     for i in range(pictureNumber):
    #         for j in range(pictureNumber):
    #             toPIL(gallery_images[pos[i][j]//batch_size][pos[i][j]%batch_size]).save(f'{path}/{name}/{i}/{j}_{gallery_targets[pos[i][j]//batch_size][pos[i][j]%batch_size].item()}.jpg')
    #             if gallery_targets[pos[i][j]//batch_size][pos[i][j]%batch_size].item()!=g_pids_save[name][i][j]:
    #                 print("target =",gallery_targets[pos[i][j]//batch_size][pos[i][j]%batch_size].item())
    #                 print('g_pid_save = ',g_pids_save[name][i][j])

    #     print('gallery set pictures are all saved')



    # save method 2
    if save_pic:
        toPIL = transforms.ToPILImage()
        for name,pos in result_order.items():
            print(f'start to save pictures of query&gallery set of {name}')
        
            for i in range(pictureNumber):
                img = query_images[i]
                target = query_pids[i]
                os.makedirs(f'{path}/{name}/{i}')
                toPIL(img).save(f'{path}/{name}/{i}/q_{target}.jpg')
                if q_pid_save[name]==None:
                    continue
                if target!=q_pid_save[name][i]:
                    print("target =",target)
                    print('q_pid_save = ',q_pid_save[name][i])
        
            print('query set pictures are all saved ')
            if pos==[]:
                continue
            for i in range(pictureNumber):
                for j in range(pictureNumber):
                    toPIL(gallery_images[pos[i][j]]).save(f'{path}/{name}/{i}/{j}_{gallery_pids[pos[i][j]].item()}.jpg')
                    if gallery_pids[pos[i][j]].item()!=g_pids_save[name][i][j]:
                        print("target =",gallery_pids[pos[i][j]].item())
                        print('g_pid_save = ',g_pids_save[name][i][j])

            print('gallery set pictures are all saved')

    else :
        print('You choose not to save pictures')

def log(cfg,q_pid_save,g_pids_save,pictureNumber):
    path = f"{os.getcwd()}/logs_pic"

    file = open(path+"/log.txt",'a')# write from the end of the txt,so it will record all your jobs
    file.write('|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n')
    time_info = time.asctime( time.localtime(time.time()))
    file.write("the log time is "+time_info+"\n\n") 

    file.write('Configuration:\n')
    file.write(f"     dataset: {cfg.DATASETS.NAMES[0]}\n")
    file.write(f"     attack : {cfg.ATTACKMETHOD!=None}\n")
    if cfg.ATTACKMETHOD!=None:
        file.write(f"            attack  method    = {cfg.ATTACKMETHOD}\n")
        file.write(f"            attack  type      = {cfg.ATTACKTYPE}\n")
        file.write(f"            attack  direction = {cfg.ATTACKDIRECTION}\n")
    file.write(f"     defense: {cfg.DEFENSEMETHOD!=None} \n")
    if cfg.DEFENSEMETHOD!=None:
        file.write(f"            defense method    = {cfg.DEFENSEMETHOD}\n"  )
    
    for name in ('origin','attack','defense'):
        rank = [0 for _ in range(pictureNumber)]
        file.write('-------------------------------------------------------\n')
        file.write(f'in {name} \n')
        file.write('-------------------------------------------------------\n')
        if q_pid_save[name]==None:
            file.write(f'no result from {name}\n')
        else:
            file.write('query pid = ')
            for i in range(pictureNumber):
                file.write(f'{q_pid_save[name][i]} ')
            file.write('\n')
            for i in range(pictureNumber):
                file.write("corresponding target :")
                for j in range(pictureNumber):
                    if g_pids_save[name][i][j]==q_pid_save[name][i]:
                        rank[j]+=1
                        file.write(' 1')
                    else:
                        file.write(' 0')
                file.write('\n')
            file.write("\n")
            for i in range(1,pictureNumber):
                rank[i]+=rank[i-1]
            for i in [1,5,10]:
                if i>pictureNumber:
                    file.write(f"rank-{i} = {rank[i]*100.0/(pictureNumber*i)}%\n")
            
            # for i in range(pictureNumber):
            #     file.write("gallery pids = ")
            #     for j in range(pictureNumber):
            #         file.write(str(g_pids_save[name][i][j])+' ')
            #     file.write('\n')
            file.write('\n')
    file.write('\nlog over\n')
    file.write('|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n')
    file.close()
    print("log over")

def print_info(str):
    print('---------------------------------------------------------')
    print('---------------------------------------------------------')
    print(str)
    print('---------------------------------------------------------')
    print('---------------------------------------------------------')

def print_configCondition(args,cfg):
    print('---------------------------------------------------------')
    print('---------------------------------------------------------')
    print('Option:')
    print(f"     attack : {args.attack!=None}")
    print(f"            attack  method    = {cfg.ATTACKMETHOD}")
    print(f"            attack  type      = {cfg.ATTACKTYPE}")
    print(f"            attack  direction = {cfg.ATTACKDIRECTION}")
    print(f"     defense: {args.defense!=None} ")
    print(f"            defense method    = {cfg.DEFENSEMETHOD}"  )
    print(f"     record : {args.record}")
    if args.record:
        print(f"            data will be recorded in ./{excel_name}")
    else:
        print(f"            data will not be recorded")

    print(f"     log    : {args.log}")
    if args.log:
        print(f"            data will be recorded in ./{excel_name}")
        if args.save_pic:
            print(f"            pictures will be saved")
        else:
            print(f"            pictures will not be saved")
    else:
        print(f"            data will not be recorded")
    print('---------------------------------------------------------')
    print('---------------------------------------------------------')