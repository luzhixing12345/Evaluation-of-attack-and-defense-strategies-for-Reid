import time
from sklearn.feature_extraction import image
import torch.nn as nn
import torch
import copy
from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.reid_patch import eval_ssim, eval_train, get_result, get_train_set, make_dict
from fastreid.utils.attack_patch.attack_algorithm import *
device='cuda'
 

class adversary_defense:
    '''
    There are two ways to do adversary defense, one way is to split the training set into query/gallery and 
    attack them as usual, the other way is to just use the key point of each attack algorithm to generate 
    the adv_image, my solution is the second way.
    '''
    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self.train_set = get_train_set(self.cfg)
        self.initialization()

    def initialization(self):
        self.cfg = DefaultTrainer.auto_scale_hyperparams(self.cfg,self.train_set.dataset.num_classes)
        self.model = DefaultTrainer.build_model_main(self.cfg)#this model was used for later evaluations
        self.model.RESIZE = True
        Checkpointer(self.model).load(self.cfg.MODEL.WEIGHTS)
        self.model.to(device)

        self.SSAE_generator = None
        self.MISR_generator = None

        if self.cfg.ATTACKMETHOD == 'SSAE':
            self.SSAE_generator = make_SSAE_generator(self.cfg,self.model,pretrained=True)
        elif self.cfg.ATTACKMETHOD == 'MISR':
            self.MISR_generator = make_MIS_Ranking_generator(self.cfg,pretrained=True)


    def get_defense_result(self):
        return get_result(self.cfg,self.cfg.MODEL.DEFENSE_TRAINED_WEIGHT,'defense')

    def defense(self):
        attack_method = self.getAttackMethod()

        optimizer = DefaultTrainer.build_optimizer(self.cfg, self.model)

        EPOCH = 3
        max_id = 2000
        alpha = 0.5 # mixing ratio between clean data and attack data,and it represents the ratio of clean data
        frequency = 800 
        loss_fun = nn.CrossEntropyLoss()

        # self.model.train()
        # for _,parm in enumerate(self.model.parameters()):
        #     parm.requires_grad=True
        # print('all parameters of model requires_grad')

        for epoch in range(EPOCH):
            loss_total = 0
            print(f'start training for epoch {epoch} of {EPOCH}')
            time_stamp_start = time.strftime("%H:%M:%S", time.localtime()) 
            self.model.train()
            for id,data in enumerate(self.train_set):
                if id>max_id:
                    break
                target = data['targets'].to(device)
                images = data['images'].to(device)
                    
                
                self.model.heads.MODE = 'C'
                adv_images = attack_method(images,target)
                
                if id % frequency==0:
                    print(f'ssim = {eval_ssim(images,adv_images)} in epoch {epoch} of {id}')
                
                #self.model.heads.MODE = 'FC'
                output_clean = self.model(images)
                output_dirty = self.model(adv_images)
                
                loss_clean = loss_fun(output_clean,target)
                loss_dirty = loss_fun(output_dirty,target)

                optimizer.zero_grad()
                loss = loss_clean*alpha+loss_dirty*(1-alpha)
                loss_total+=loss.item()
                
                loss.backward()
                optimizer.step()
            eval_train(self.model,self.train_set)
            time_stamp_end = time.strftime("%H:%M:%S", time.localtime()) 
            print(f'total_loss for epoch {epoch} of {EPOCH} is {loss_total} | {time_stamp_start} - {time_stamp_end}')

            
        print('finished adv_training !')
        print('--------------------------------')
        Checkpointer(self.model,'model').save(f'{self.cfg.DEFENSEMETHOD}_{self.cfg.ATTACKMETHOD}_{self.cfg.DATASETS.NAMES[0]}_{self.cfg.CFGTYPE}')
        print('Successfully saved the adv_trained model !')


    def getAttackMethod(self):
        if self.cfg.ATTACKMETHOD == 'FNA':
            raise "FNA doesn't have advsary defense, use GOAT defense instead"
        #mse = nn.MSELoss(reduction='sum')
        loss_fn = nn.CrossEntropyLoss()
        def odfa(f1,f2):
            return loss_fn(-f1,f2)

        eps=0.05
        eps_iter=1.0/255.0
        
        origin_model = copy.deepcopy(self.model)
        dict = {
            'FGSM'    :FGSM  (self.cfg,origin_model, loss_fn, eps=eps, targeted=False),
            'IFGSM'   :IFGSM (self.cfg,origin_model, loss_fn, eps=eps, eps_iter=eps_iter,targeted=False,rand_init=False),
            'MIFGSM'  :MIFGSM(self.cfg,origin_model, loss_fn, eps=eps, eps_iter=eps_iter,targeted=False,decay_factor=1),
            'ODFA'    :ODFA  (self.cfg,origin_model, odfa,eps=eps, eps_iter=eps_iter,targeted=True,rand_init=False),
            'SSAE'    :self.SSAE_generator,
            'MISR'    :self.MISR_generator,
            'MUAP'    :MUAP(self.cfg,origin_model)
        }
        return dict[self.cfg.ATTACKMETHOD]


    