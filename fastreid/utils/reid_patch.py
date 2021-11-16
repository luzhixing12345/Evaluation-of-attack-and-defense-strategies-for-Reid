
import enum
import os
from sys import prefix
import openpyxl
import torch
import torch.nn as nn
from fastreid.data.build import (build_reid_test_data_loader,
                                 build_reid_query_data_loader,
                                 build_reid_att_query_data_loader,
                                 build_reid_gallery_data_loader,
                                 build_reid_train_loader)
from fastreid.engine import DefaultTrainer
from fastreid.solver.build import build_optimizer
from fastreid.utils.checkpoint import Checkpointer
from openpyxl.utils import get_column_letter
from torchvision import transforms
from skimage import transform 
import skimage
import skimage.io as io

device= 'cuda'
excel_name = 'result.xlsx'

C_Attack_algorithm_library=["FGSM",'IFGSM','MIFGSM','ODFA']  #针对分类问题的攻击算法库
R_Attack_algorithm_library=['SMA','FNA','MUAP','SSAE','ES','GTM', 'GTT', 'TMA', 'LTM','CA+','CA-','QA+','QA-','SPQA']
Attack_algorithm_library=C_Attack_algorithm_library+R_Attack_algorithm_library

G_Defense_algorithm_library=['ADV_DEF','GRA_REG','DISTILL']
R_Defense_algorithm_library=['GOAT','EST','SES','PNP']
Defense_algorithm_library=G_Defense_algorithm_library+R_Defense_algorithm_library

evaluation_indicator=['Rank-1','Rank-5','Rank-10','mAP','mINP','metric']
evaluation_attackIndex=['mAP','1-SSIM','index']
num=len(evaluation_indicator)

def get_query_set(cfg):
    query_set=build_reid_query_data_loader(cfg,cfg.DATASETS.TESTS[0])
    return query_set

def get_att_query_set(cfg):
    adv_query_set = build_reid_att_query_data_loader(cfg,cfg.DATASETS.TESTS[0])
    return adv_query_set

def get_test_set(cfg):
    test_set = build_reid_test_data_loader(cfg,cfg.DATASETS.TESTS[0])
    return test_set

def get_gallery_set(cfg):
    gallery_set=build_reid_gallery_data_loader(cfg,cfg.DATASETS.TESTS[0])
    return gallery_set

def get_train_set(cfg):
    train_data_loader = build_reid_train_loader(cfg)
    return train_data_loader

def evaluate_ssim(cfg):
    query_data_loader=get_query_set(cfg)
    att_query_data_loader=get_att_query_set(cfg)
    SSIM=0
    for data1,data2 in zip(query_data_loader,att_query_data_loader): 
        path1 = data1['img_paths']
        path2 = data2['img_paths']
        for pic1,pic2 in zip(path1,path2):
            image1 = io.imread(pic1)   
            image2 = io.imread(pic2)  
            image2=transform.resize(image2,(image1.shape))
            image2*=255
            SSIM+=skimage.measure.compare_ssim(image1,image2,multichannel=True,data_range=255)
    SSIM/=(len(query_data_loader)*cfg.TEST.IMS_PER_BATCH)
    return SSIM


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
    
    for img,path in zip(imgs,paths):  #zip 并行遍历
        if path.find('bounding_box')!=-1:
            print('save all the query set picture')
            return
        pic = toPIL(img)
        position = path.find('query/') # path给的是绝对位置
        name = path[position+6:] #只需要提取出其名字即可
        pic.save(path[:position]+str+'/'+name)
    print(f"successful save in {path[:position]+str}")

def classify_test_set(cfg,query_data_loader):
    '''
    固定住backbone的权重，利用query_set重新训练cls_layer的weight,使其也能准确的分类图像和id
    转换成分类问题，可较好的接收针对于分类问题的算法攻击
    
    '''
    #set the number of neurons of the classifier layer as query set's targets number
    cfg = DefaultTrainer.auto_scale_hyperparams(cfg,query_data_loader.dataset.num_classes)

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
        for _,data in enumerate(query_data_loader):
            targets = data['targets'].to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = loss_fun(logits,targets)
            loss.backward()
            optimizer.step()
            
        accurency = eval_train(model,query_data_loader) #show the accurrency
        print('The accurency of query set in Train Epoch {} is {}'.format(i,accurency))
        print('---------------------------------------------------------------------')
    Checkpointer(model,'model').save('test_trained') # model was saved in ./model/test_trained.pth
    print('the model was saved in ./model/test_trained.pth')

def match_type(cfg,type):
    
    if type == 'attack':
        atk_method=cfg.MODEL.ATTACKMETHOD
        if atk_method in C_Attack_algorithm_library:
            return True
        elif atk_method in R_Attack_algorithm_library:
            return False
        else:
            raise KeyError('you should use the attack method in the library, or check your spelling')
    else:
        def_method=cfg.MODEL.DEFENSEMETHOD
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
        
        
        images = images * 255.0
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
    for row in range (3,4+len(Attack_algorithm_library)):
        for col in range (2,6):
            sheet_AI.column_dimensions[get_column_letter(col)].width = 20
            sheet_AI.row_dimensions[row].height = 40
    
    for row,name in enumerate(Attack_algorithm_library):
        sheet_AI['B'+str(4+row)]=name
    
    for i in range(len(evaluation_attackIndex)):
        sheet_AI[get_column_letter(i+3)+'3'] = evaluation_attackIndex[i]

def save_data(cfg,pure_result,att_result,def_result,def_adv_result,sheet):

    row_start = 4 + num*Attack_algorithm_library.index(cfg.MODEL.ATTACKMETHOD)
    column1 = chr(ord('F') + Defense_algorithm_library.index(cfg.MODEL.DEFENSEMETHOD)*2)
    column2 = chr(ord(column1)+1)

    for col,dict_name in {'D':pure_result,'E':att_result,column1:def_result,column2:def_adv_result}.items():
        for i in range(num):
            sheet[col+str(row_start+i)]  =dict_name[evaluation_indicator[i]]

def save_attack_index(cfg,pure_result,att_result,SSIM,sheet):
    row_start = 4 + Attack_algorithm_library.index(cfg.MODEL.ATTACKMETHOD)
    column = 3
    
    index = {}
    index['mAP'] = pure_result['mAP']-att_result['mAP']
    index['SSIM'] = 1-SSIM
    index['index'] = index['mAP']/index['SSIM']

    for i in range(len(evaluation_attackIndex)):
        sheet[get_column_letter(column+i)+str(row_start)] = index[evaluation_attackIndex[i]]


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
    if def_result!=None and def_adv_result != None:
        save_data(cfg,pure_result,att_result,def_result,def_adv_result,sheet)

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

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path+"/origin") 
        os.makedirs(path+"/attack") 
        os.makedirs(path+"/defense") 
        print("dirtory "+path+" has been created")
    else:
        return


def record_order(cfg,pure_result,query_set,gallery_set):
    result_order = pure_result['result_order']
    q_pid_save   = pure_result['q_pid_save']
    g_pids_save  = pure_result['g_pids_save']
    path = os.getcwd()+'/logs_pic/'+cfg.DATASETS.NAMES[0]+'/'+cfg.MODEL.ATTACKMETHOD+'_'+cfg.MODEL.DEFENSEMETHOD
    mkdir(path)
    toPIL = transforms.ToPILImage()
    for idx,data in enumerate(query_set):
        img = data['images'][0].cpu()/255
        target = data['targets'][0].cpu()
        toPIL(img).save(path+'/'+str(target.item())+'.jpg')
        if target.item()==q_pid_save:
            print("yes!!!!")
        else :
            print("target =",target)
            print('q_pid_save = ',q_pid_save)
        break
    
    batch_size = cfg.TEST.IMS_PER_BATCH
    gallery_img = []
    gallery_target = []
    for idx,data in enumerate(gallery_set):
        img = data['images'].cpu()/255
        target = data['targets'].cpu()
        gallery_img.append(img)
        gallery_target.append(target)
    
    s=0
    for id in result_order:
        toPIL(gallery_img[id//batch_size][id%batch_size]).save(path+'/origin/'+str(s)+'_'+str(gallery_target[id//batch_size][id%batch_size].item())+'.jpg')
        if gallery_target[id//batch_size][id%batch_size].item()==g_pids_save[s]:
            print('ohhhhh')
        else:
            print(gallery_img[id//batch_size][id%batch_size].item())
            print(g_pids_save[s])
        s+=1
    