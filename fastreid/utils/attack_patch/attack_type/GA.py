

import copy
import torch
import torch.nn as nn
import numpy as np
import skimage
from fastreid.engine import DefaultTrainer
from fastreid.modeling.heads.build import build_feature_heads
from fastreid.utils.attack_patch.attack_algorithm import *
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.reid_patch import CHW_to_HWC, change_preprocess_image, get_query_set


from fastreid.utils.compute_dist import build_dist
device = 'cuda'

class GalleryAttack:
    '''
    do GA+- for retrieval algorithm
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
        evaluation_indicator=['Rank-1','Rank-5','Rank-10','mAP']
        self.rank={}
        for indicator in evaluation_indicator:
            self.rank[indicator]=0
            
        self.SSAE_generator = None
        self.MISR_generator = None
        if self.cfg.ATTACKMETHOD=='SSAE':
            self.SSAE_generator = make_SSAE_generator(self.cfg,self.model,pretrained=self.pretrained)
        elif self.cfg.ATTACKMETHOD=='MISR':
            self.MISR_generator = make_MIS_Ranking_generator(self.cfg,self.model,ak_type=-1,pretrained=self.pretrained)

    def pretreatment(self):
        test_dataset,self.num_query = DefaultTrainer.build_test_loader(self.cfg,dataset_name=self.cfg.DATASETS.NAMES[0])
        images = []
        pids = []
        camids = []
        features = []

        for _,data in enumerate(test_dataset):
            images.append((data['images']/255.0).cpu())
            with torch.no_grad():
                features.append(self.model(data['images']/255.0).cpu())
            pids.append(data['targets'].cpu())
            camids.append(data['camids'].cpu())
        
        images = torch.cat(images,dim=0)
        features = torch.cat(features, dim=0)
        pids = torch.cat(pids, dim=0).numpy()
        camids = torch.cat(camids,dim=0).numpy()

        self.query_images = images[:self.num_query]
        self.query_pids = pids[:self.num_query]
        self.query_camids = camids[:self.num_query]
        self.query_features = features[:self.num_query]

        self.gallery_images = images[self.num_query:]
        self.gallery_pids = pids[self.num_query:]
        self.gallery_camids = camids[self.num_query:]
        self.gallery_features = features[self.num_query:]

        self.dist = build_dist(self.query_features, self.gallery_features, self.cfg.TEST.METRIC)

        self.indices = np.argsort(self.dist, axis=1)
        self.matches = (self.gallery_pids[self.indices] == self.query_pids[:, np.newaxis]).astype(np.int32)#3368x15913
        
    def get_attack_method(self):
        
        mse = nn.MSELoss(reduction='sum')
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

        eps=0.05
        eps_iter=1.0/255.0
        dict = {
            'R-FGSM'  :FGSM(self.cfg,self.model, mse, eps=eps,targeted=self.target),
            'R-IFGSM' :IFGSM(self.cfg,self.model,mse, eps=eps, eps_iter=eps_iter,targeted=self.target,rand_init=False),
            'R-MIFGSM':MIFGSM(self.cfg,self.model,mse, eps=eps, eps_iter=eps_iter,targeted=self.target,decay_factor=1),
            'ODFA'    :ODFA(self.cfg,self.model, odfa, eps=eps, eps_iter=eps_iter,targeted=not self.target,rand_init=False),
            'SMA'     :IFGSM(self.cfg,self.model, mse, eps=eps, eps_iter=eps_iter ,targeted=self.target,rand_init=False),
            'FNA'     :IFGSM(self.cfg,self.model, max_min_mse, eps=eps, eps_iter=eps_iter,targeted=self.target,rand_init=False),
            'SSAE'    :self.SSAE_generator,
            'MISR'    :self.MISR_generator,
            'MUAP'    :MUAP(self.cfg,self.model)
        }
        return dict[self.cfg.ATTACKMETHOD]

    def attack_images(self,features,selected_images,target):

        attack_method = self.get_attack_method()
        if self.cfg.ATTACKMETHOD =='MUAP':
            new_selected_samples = attack_method(selected_images,target)
        else:
            new_selected_samples = attack_method(selected_images,features)

        return new_selected_samples

    def select_samples(self,id,size,direction):
        selected_images = []
        selected_ids = []
        id = id*self.batch_size
        # use size instead of batch size is because query set may not be divisible by batch_size and there
        # may be a few images left as remainder

        for i in range(size):
            q_idx = id+i
            q_pid = self.query_pids[q_idx]
            q_camid = self.query_camids[q_idx]

            order = self.indices[q_idx]#15913
            remove = (self.gallery_pids[order] == q_pid) & (self.gallery_camids[order] == q_camid)        
            keep = np.invert(remove)

            if direction=='+':
                # random sampling from populationfor GA+
                sample_pos = torch.randint(len(self.indices[q_idx][keep]), (10,))  # [output_0,M]
                sample_id  = self.indices[q_idx][keep][sample_pos]

                selected_images.append(self.gallery_images[sample_id])
                selected_ids.append(torch.tensor(sample_id))
            else :
                # random sampling from top-3M for GA-
                sample_id = self.indices[q_idx][keep][:10]

                selected_images.append(self.gallery_images[sample_id])
                selected_ids.append(torch.tensor(sample_id))

        selected_images = torch.stack(selected_images)
        selected_ids  =  torch.stack(selected_ids)
        return selected_images,selected_ids

    def evaluate(self,id,image_features,selected_ids,selected_images,new_selected_images):
        
        size,N= selected_ids.shape
        id = id*self.batch_size
        # use size instead of batch size is because query set may not be divisible by batch_size and there
        # may be a few images left as remainder
        SSIM = 0
        for i in range(size):
            for j in range(N):
                image1 = CHW_to_HWC(selected_images[i][j])
                image2 = CHW_to_HWC(new_selected_images[i][j])
                SSIM += skimage.measure.compare_ssim(image1,image2,multichannel=True)
        
        SSIM/=size*N
        self.SSIM += SSIM

        for i in range(size):
            q_idx = id+i
            # deepcopy gallery features and replace the corresponding postion with the after-attack gallery images
            gallery_features = copy.deepcopy(self.gallery_features)
            # gallery features(size) = (gallery imaegs number) x 2048
            # new selected images(size) = N(10) x batch_size x 3 x 256 x 128
            # selected ids(size) = batch_size x N(10)
            for j in range(N):
                with torch.no_grad():
                    features = self.model(new_selected_images[:,j,:,:,:])
                #features(size) = size x 2048
                gallery_features[selected_ids[i][j]]=features[i]

            query_features = image_features[i].cpu().unsqueeze(0)
            dist = build_dist(query_features,gallery_features,self.cfg.TEST.METRIC)
            order = np.argsort(dist, axis=1)
            matches = (self.gallery_pids[order] == self.query_pids[q_idx]).astype(np.int32)
            remove = (self.gallery_pids[order] == self.query_pids[q_idx]) & (self.gallery_camids[order] == self.query_camids[q_idx])        
            keep = np.invert(remove)

            raw_cmc = matches[keep]

            cmc = raw_cmc.cumsum()
            num_rel = raw_cmc.sum()
            tmp_cmc = raw_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
            AP = tmp_cmc.sum() / num_rel

            for r in [1, 5, 10]:
                self.rank['Rank-{}'.format(r)] += cmc[r - 1]>=1
            self.rank['mAP']+=AP


    def get_result(self):
        for indictor in self.rank.keys():
            self.rank[indictor]=self.rank[indictor]*100.0/self.num_query

        self.SSIM/=len(self.query_data_loader)

        print(self.rank)
        print('SSIM = ',self.SSIM)
        return self.rank,self.SSIM
        

    def attack(self):
        
        self.query_data_loader = get_query_set(self.cfg)
        for q_idx ,data in enumerate(self.query_data_loader):
            
            images = (data['images']/255).clone().to(device).detach()
            target = data['targets'].to(device)

            selected_images,selected_ids= self.select_samples(q_idx,images.shape[0],self.cfg.ATTACKDIRECTION)
            with torch.no_grad():
                features = self.model(images)
            new_selected_images = self.attack_images(features,selected_images,target)

            self.evaluate(q_idx,features,selected_ids,selected_images,new_selected_images)

            

