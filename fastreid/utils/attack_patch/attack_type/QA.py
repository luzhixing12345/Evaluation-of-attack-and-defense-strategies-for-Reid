
import torch
import torch.nn as nn
import numpy as np
from fastreid.engine import DefaultTrainer
from fastreid.modeling.heads.build import build_feature_heads
from fastreid.utils.attack_patch.attack_algorithm import *
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.reid_patch import CHW_to_HWC, change_preprocess_image, get_query_set, get_result, save_image
import skimage

from fastreid.utils.compute_dist import build_dist

device = 'cuda'
class QueryAttack:
    '''
    do QA+- for retrieval algorithm
    '''
    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self.batch_size = cfg.TEST.IMS_PER_BATCH
        self.direction  = cfg.ATTACKDIRECTION
        self.pretrained = cfg.ATTACKPRETRAINED
        self.target = True if self.direction=='+' else False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  
        self.default_setup()
        self.pretreatment()

    def default_setup(self):
        
        self.model = DefaultTrainer.build_model_main(self.cfg)#this model was used for later evaluations
        self.model.preprocess_image = change_preprocess_image(self.cfg) 
        self.model.heads = build_feature_heads(self.cfg)
        self.model.to(self.device)
        Checkpointer(self.model).load(self.cfg.MODEL.WEIGHTS)

        self.SSIM=0
        self.SSAE_generator = None
        self.MISR_generator = None
        if self.cfg.ATTACKMETHOD=='SSAE':
            self.SSAE_generator = make_SSAE_generator(self.cfg,self.model,pretrained=self.pretrained)
        elif self.cfg.ATTACKMETHOD=='MISR':
            self.MISR_generator = make_MIS_Ranking_generator(self.cfg,self.model,ak_type=-1,pretrained=self.pretrained)
    
    def pretreatment(self):
        test_dataset,num_query = DefaultTrainer.build_test_loader(self.cfg,dataset_name=self.cfg.DATASETS.NAMES[0])
        pids = []
        camids = []
        features = []

        for _,data in enumerate(test_dataset):
            with torch.no_grad():
                features.append(self.model(data['images']/255.0).cpu())
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
        
    def attack_images(self,images,selected_features,target):

        attack_method = self.get_attack_method()
        if self.cfg.ATTACKMETHOD=='ODFA':
            with torch.no_grad():
                features = self.model(images)
            adv_images = attack_method(images,features)
        elif self.cfg.ATTACKMETHOD=='MUAP':
            adv_images = attack_method(images,target)
        else :
            adv_images = attack_method(images,selected_features)

        return adv_images

    def select_samples(self,id,size,direction):
        selected_features = []
        id = id*self.batch_size

        for i in range(size):
            q_idx = id+i
            q_pid = self.query_pids[q_idx]
            q_camid = self.query_camids[q_idx]

            order = self.indices[q_idx]#15913
            remove = (self.gallery_pids[order] == q_pid) & (self.gallery_camids[order] == q_camid)        
            keep = np.invert(remove)

            if direction=='+':
                # random sampling from populationfor QA+
                sample_pos = torch.randint(len(self.indices[q_idx][keep]), (10,))  # [output_0,M]
                sample_id  = self.indices[q_idx][keep][sample_pos]

                selected_features.append(self.gallery_features[sample_id])
            else :
                # random sampling from top-3M for QA-
                sample_id = self.indices[q_idx][keep][:10]

                selected_features.append(self.gallery_features[sample_id])

        selected_features = torch.stack(selected_features)
        return selected_features

    def get_attack_method(self):
        
        mse = nn.MSELoss(reduction='sum')
        def QA_MSE(f1s, f2s):
            m = 0
            for f1, f2 in zip(f1s, f2s):
                for f in f2:
                    m += mse(f1,f)
            return m
        def odfa(f1, f2):
            return mse(f1,-f2)
        def max_min_mse(f1s, f2s):
            # f1, f2 = fs[:,0,:], fs[:,1,:]
            # return mse(x,f1) - mse(x,f2)
            m = 0
            for f1, f2 in zip(f1s, f2s):
                for i in range(len(f2)-1):
                    m += mse(f1,f2[i])
                m -= mse(f1,f2[-1])
            return m

        eps=0.05
        eps_iter=1.0/255.0
        dict = {
            'R-FGSM'  :FGSM  (self.cfg,self.model, QA_MSE, eps=eps,targeted=self.target),
            'R-IFGSM' :IFGSM (self.cfg,self.model, QA_MSE, eps=eps, eps_iter=eps_iter,targeted=self.target,rand_init=False),
            'R-MIFGSM':MIFGSM(self.cfg,self.model, QA_MSE, eps=eps, eps_iter=eps_iter,targeted=self.target,decay_factor=1),
            'ODFA'    :ODFA  (self.cfg,self.model, odfa,   eps=eps, eps_iter=eps_iter,targeted=True,rand_init=False),
            'SMA'     :IFGSM (self.cfg,self.model, mse,    eps=eps, eps_iter=eps_iter ,targeted=self.target,rand_init=False),
            'FNA'     :IFGSM (self.cfg,self.model, max_min_mse, eps=eps, eps_iter=eps_iter,targeted=self.target,rand_init=False),
            'SSAE'    :self.SSAE_generator,
            'MISR'    :self.MISR_generator,
            'MUAP'    :MUAP(self.cfg,self.model)
        }
        return dict[self.cfg.ATTACKMETHOD]

    def evaluate(self,images1,images2):
        size = images1.shape[0]
        SSIM = 0
        for i in range(size):
            image1 = CHW_to_HWC(images1[i])
            image2 = CHW_to_HWC(images2[i])
            SSIM += skimage.measure.compare_ssim(image1,image2,multichannel=True)
        SSIM/=size
        self.SSIM+=SSIM


    def attack(self):
        
        self.query_data_loader = get_query_set(self.cfg)
        for q_idx ,data in enumerate(self.query_data_loader):
            
            images = (data['images']/255).clone().to(device).detach()
            target = data['targets'].to(device)
            path = data['img_paths']
            #images_orig = images.clone()
            images = images.requires_grad_()

            selected_features = self.select_samples(q_idx,images.shape[0],self.cfg.ATTACKDIRECTION)
            
            adv_images = self.attack_images(images,selected_features,target)
            self.evaluate(images,adv_images)
            
            save_image(adv_images,path,'adv_query')
        
    def get_result(self):
        self.SSIM/=len(self.query_data_loader)
        return get_result(self.cfg,self.cfg.MODEL.WEIGHTS,'attack'),self.SSIM
    

