
#这里引用了github的项目advtorch的库
#如想详细了解请阅读源代码
#https://github.com/BorealisAI/advertorch
from enum import EnumMeta
from advertorch.context import ctx_noparamgrad_and_eval
from numpy.lib.function_base import select
from skimage.util.dtype import img_as_bool
from fastreid.utils.reid_patch import evaluate_ssim, get_gallery_set, get_query_set, release_cuda_memory, save_image,get_result,classify_test_set,Attack_algorithm_library,change_preprocess_image
from fastreid.modeling.heads import build_feature_heads
from fastreid.utils.advertorch.attacks import FGSM,MomentumIterativeAttack,LinfPGDAttack,CarliniWagnerL2Attack
import torch
from fastreid.utils.attack_patch.SMA import SMA
from fastreid.utils.attack_patch.FNA import FNA
from fastreid.utils.attack_patch.MIS_RANKING import mis_ranking
from fastreid.utils.attack_patch.MUAP import MUAP
#from fastreid.utils.attack_patch import cw  
from fastreid.utils.attack_patch.SSAE import SSAE_attack
from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer
import torch.nn.functional as F
import numpy as np
from fastreid.utils.compute_dist import build_dist

device = 'cuda'
attack_type = ['QA','GA']
# C_Attack_algorithm_library=["FGSM",'IFGSM','MIFGSM','CW','ODFA']  #针对分类问题的攻击算法库
# #DDNL2 SPA CW时间太长了！！！
# R_Attack_algorithm_library=['SMA','FNA','MIS-RANKING']
# Attack_algorithm_library=C_Attack_algorithm_library+R_Attack_algorithm_library

def attack(cfg,query_data_loader,gallery_data_loader,type,model_path='./model/test_trained.pth'):
    
    if type:  # 针对分类问题的攻击
        if cfg.ATTACKTYPE=='QA':
            data_loader = query_data_loader
        
        elif cfg.ATTACKTYPE=='GA':
            data_loader = gallery_data_loader
            
        else:
            raise KeyError(f"attack type must be in {attack_type}")
        classify_test_set(cfg, data_loader)
        # 设计分类器，能够完成query_set的image到target的对应,没有完成gallery_set的对应
        # 模型保存的位置在(./model/test_trained.pth)其对query_set上的数据有较高的识别率，但对原train_set识别率下降
        # 测试了训练query_set使其达到较高准确率后对原train_set标签对应的影响,大概准确率降至20%
        # 最后得到两个模型
        #  cfg.MODEL.WEIGHTS has a high accurency in train-set
        #  cfg.MODEL.TESTSET_TRAINED_WEIGHT has a high accurency in query_set
        # the two models both perform bad in opposite set,so do not mis-use .
        print('finished the train for test set ')
        print('---------------------------------------------')
        release_cuda_memory()
        att_result,att_result_to_save = attack_C(cfg,data_loader,model_path)
    else:  # 针对排序问题的攻击
        attack_R(cfg, query_data_loader,gallery_data_loader)
        att_result,att_result_to_save = get_result(cfg,cfg.MODEL.WEIGHTS,'attack')

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


def match_attack_method(cfg,model,data_loader):
    mse = torch.nn.MSELoss(reduction='sum')
    def max_min_mse(f1s, f2s):
        m = 0
        for f1, f2 in zip(f1s, f2s):
            for i in range(len(f2)-1):
                m += mse(f1,f2[i])
            m -= mse(f1,f2[-1])
        return m
    
    atk_method=cfg.ATTACKMETHOD
    #分类问题攻击算法
    if atk_method=='FGSM':
        return FGSM(model,eps=0.05,targeted=False)
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


class QA:
    def __init__(self,cfg,model,model_rank) -> None:
        self.cfg = cfg
        self.batch_size = cfg.TEST.IMS_PER_BATCH
        self.model = model            
        self.model_rank = model_rank
        self.EPOCH = 24
        self.pretreatment()
        
        
    def pretreatment(self):
        test_dataset,num_query = DefaultTrainer.build_test_loader(self.cfg,dataset_name=self.cfg.DATASETS.NAMES[0])
        pids = []
        camids = []
        features = []

        for _,data in enumerate(test_dataset):
            with torch.no_grad():
                features.append(self.model_rank(data['images']/255.0).cpu())
            pids.append(data['targets'].cpu())
            camids.append(data['camids'].cpu())
        
        features = torch.cat(features, dim=0)
        pids = torch.cat(pids, dim=0).numpy()
        camids = torch.cat(camids,dim=0).numpy()

        self.query_pids = pids[:num_query]
        self.query_camids = camids[:num_query]
        self.query_features = features[:num_query]

        self.gallery_pids = pids[num_query:]
        self.gallery_camids = camids[num_query:]
        self.gallery_features = features[num_query:]

        self.dist = build_dist(self.query_features, self.gallery_features, self.cfg.TEST.METRIC)

        self.indices = np.argsort(self.dist, axis=1)
        self.matches = (self.gallery_pids[self.indices] == self.query_pids[:, np.newaxis]).astype(np.int32)#3368x15913
        

    def attack_images(self,images):
        adversary=match_attack_method(self.cfg,self.model,None)
        with ctx_noparamgrad_and_eval(self.model):
            images = adversary.perturb(images)  
        return images.clone().detach()

    def get_rank_loss(self, features: torch.Tensor, selected_samples: torch.Tensor,
                            *, direction: str, dist: torch.Tensor = None, cidx: torch.Tensor = None):
        '''
        Computes the loss function for pure query attack

        Arguments:
            features: size(batch, embedding_dim), query embeddings.
            selected_samples: size(batch, M, embedding_dim), selected candidates.
            self.gallery_features: size(testsize, embedding_dim), embedding of test set.
            direction: either '+' or '-'.
            dist: size(batch, testsize), pairwise distance matrix.
            cidx: size(batch, M), index of candidates in self.gallery_features.
        '''
        # features,shape = torch.Size([32, 2048])
        # selected_samples =  torch.Size([32, 10, 2048])
        # self.gallery_features =  torch.Size([17661, 2048])

        assert(features.shape[1] == selected_samples.shape[2] == self.gallery_features.shape[1])
        batch_size = features.shape[0]
        M  = selected_samples.shape[1]
        D  = selected_samples.shape[2] 
        NX = self.gallery_features.shape[0]
        DO_RANK = (dist is not None) and (cidx is not None)
        losses, ranks = [], []
        #refrank = []
        for i in range(batch_size):
            # == compute the pairwise loss
            q = features[i].view(1, D)  # [1, output_1]
            G = selected_samples[i, :, :].view(M, D)  # [1, output_1]

            A = torch.cdist(q, G).view(1, M)
            B = torch.cdist(self.gallery_features, q).view(NX, 1)
            # [XXX] the old method suffer from large memory footprint
            # A = (C - q).norm(2, dim=1).view(1, M)
            # B = (self.gallery_features - q).norm(2, dim=1).view(NX, 1)
            # == loss function
            if '+' == direction:
                loss = (A - B).clamp(min=0.).mean()
            elif '-' == direction:
                loss = (-A + B).clamp(min=0.).mean()
            losses.append(loss)
            # == compute the rank
            if DO_RANK:
                ranks.append(torch.mean(dist[i].flatten().argsort().argsort()
                                     [cidx[i, :].flatten()].float()).item())
            #refrank.append( ((A>B).float().mean()).item() )
        #print('(debug)', 'rank=', statistics.mean(refrank))
        loss = torch.stack(losses).mean()
        return loss
    

    def select_samples(self,id,size,direction):
        selected_samples = []
        id = id*self.batch_size

        for i in range(size):
            q_idx = id+i
            q_pid = self.query_pids[q_idx]
            q_camid = self.query_camids[q_idx]

            order = self.indices[q_idx]#15913
            remove = (self.gallery_pids[order] == q_pid) & (self.gallery_camids[order] == q_camid)        
            keep = np.invert(remove)

            if direction=='+':
                M = 1
                # random sampling from populationfor QA+
                sample_id = torch.randint(len(self.indices[q_idx][keep]), (10,))  # [output_0,M]

                selected_samples.append(self.gallery_features[keep][sample_id])
            else :
                M = 1
                # random sampling from top-3M for QA-
                sample_id = self.indices[q_idx][keep][:10]

                selected_samples.append(self.gallery_features[keep][sample_id]
                )
        selected_samples = torch.stack(selected_samples)
        return selected_samples

    def attack(self):
        alpha = 1/255.0
        query_data_loader = get_query_set(self.cfg)
        
        for q_idx ,data in enumerate(query_data_loader):
            
            images = (data['images']/255).clone().detach()
            path = data['img_paths']
            print('images.shape = ',images.shape)
            #images_orig = images.clone()
            images = images.requires_grad_()

            selected_samples = self.select_samples(q_idx,images.shape[0],self.cfg.ATTACKDIRECTION)
            
            images = self.attack_images(images)
            images = images.requires_grad_()
            
            for iteration in range(self.EPOCH):
            # >> prepare optimizer for SGD
                optim = torch.optim.SGD(self.model.parameters(), lr=1.)
                optimx = torch.optim.SGD([images], lr=1.)
                optim.zero_grad()
                optimx.zero_grad()
                features = self.model_rank(images)
                loss = self.get_rank_loss(features.cpu(), selected_samples,direction=self.cfg.ATTACKDIRECTION)

                loss.backward(retain_graph = True)
                
                images.grad.data.copy_(alpha * torch.sign(images.grad))
    
                optimx.step()
                # L_infty constraint
                #images = torch.min(images, images_orig + self.eps)
                # L_infty constraint
                #images = torch.max(images, images_orig - self.eps)
                images = torch.clamp(images, min=0., max=1.)
                images = images.clone().detach()
                images.requires_grad = True

            optim.zero_grad()
            optimx.zero_grad()

            save_image(images,path,'adv_query')

class GA:
    def __init__(self,cfg,model,model_rank) -> None:
        self.cfg = cfg
        self.model = model            
        self.model_rank = model_rank
        self.EPOCH = 24
        self.pretreatment()
        
    def pretreatment(self):
        test_dataset,num_query = DefaultTrainer.build_test_loader(self.cfg,dataset_name=self.cfg.DATASETS.NAMES[0])
        images = []
        pids = []
        camids = []
        features = []

        for _,data in enumerate(test_dataset):
            images.append((data['images']/255.0).cpu())
            with torch.no_grad():
                features.append(self.model_rank(data['images']/255.0).cpu())
            pids.append(data['targets'].cpu())
            camids.append(data['camids'].cpu())
        
        images = torch.cat(images,dim=0)
        features = torch.cat(features, dim=0)
        pids = torch.cat(pids, dim=0).numpy()
        camids = torch.cat(camids,dim=0).numpy()

        self.query_images = images[:num_query]
        self.query_pids = pids[:num_query]
        self.query_camids = camids[:num_query]
        self.query_features = features[:num_query]

        self.gallery_images = images[num_query:]
        self.gallery_pids = pids[num_query:]
        self.gallery_camids = camids[num_query:]
        self.gallery_features = features[num_query:]

        self.dist = build_dist(self.query_features, self.gallery_features, self.cfg.TEST.METRIC)

        self.indices = np.argsort(self.dist, axis=1)
        self.matches = (self.gallery_pids[self.indices] == self.query_pids[:, np.newaxis]).astype(np.int32)#3368x15913
 

    def select_samples(self,q_idx,direction):
        selected_samples = []

        for i in range(self.batch_size):
            q_idx = q_idx*self.batch_size+i
            q_pid = self.query_pids[q_idx]
            q_camid = self.query_camids[q_idx]

            order = self.indices[q_idx]#15913
            remove = (self.gallery_pids[order] == q_pid) & (self.gallery_camids[order] == q_camid)        
            keep = np.invert(remove)

            if direction=='+':
                M = 1
                # random sampling from populationfor QA+
                sample_id = torch.randint(len(self.indices[q_idx][keep]), (10,))  # [output_0,M]

                selected_samples.append(self.gallery_features[keep][sample_id].to(device))
            else :
                M = 1
                # random sampling from top-3M for QA-
                sample_id = self.indices[q_idx][keep][:10]

                selected_samples.append(self.gallery_features[keep][sample_id].to(device))
        selected_samples = torch.stack(selected_samples)
        print('selected_sample.shape = ',selected_samples.shape)
        return selected_samples



    def get_dist(self,images):
        with torch.no_grad():
            features = self.model_rank(images)
            dist = torch.cdist(features, self.gallery_features)
            dist_detach = dist.clone().detach()
        return dist_detach 
    
    def attack(self):
        alpha = 1/255.0
        query_data_loader = get_query_set(self.cfg)
        for _,data in enumerate(query_data_loader):
            images = (data['images']/255.0).to(device)
            images = images.detach()
            #images_orig = images.clone()
            images = images.requires_grad_()

            dist = self.get_dist(images)
            selected_samples = self.select_samples(dist,self.cfg.ATTACKDIRECTION)

            images = self.attack_images(images)
            images = images.requires_grad_()
    
