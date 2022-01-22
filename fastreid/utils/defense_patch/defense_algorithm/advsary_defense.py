import time
import torch.nn as nn
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

        self.temp_cfg = copy.deepcopy(self.cfg)
        self.temp_cfg.ATTACKTYPE = 'QA'
        
        self.temp_model = copy.deepcopy(self.model)
        self.temp_model.heads.MODE = 'C'

        if self.cfg.ATTACKMETHOD == 'SSAE':
            self.SSAE_generator = make_SSAE_generator(self.temp_cfg,self.temp_model,pretrained=True)
        elif self.cfg.ATTACKMETHOD == 'MISR':
            self.MISR_generator = make_MIS_Ranking_generator(self.temp_cfg,pretrained=True)


    def get_defense_result(self):
        return get_result(self.cfg,self.cfg.MODEL.DEFENSE_TRAINED_WEIGHT,'defense')

    def defense(self):
        attack_method = self.getAttackMethod()

        optimizer = DefaultTrainer.build_optimizer(self.cfg, self.model)
        #loss_fun = nn.CrossEntropyLoss()

        EPOCH = 3
        max_id = 4000
        alpha = 0.5 # mixing ratio between clean data and attack data,and it represents the ratio of clean data
        frequency = 800 

        self.model.train()
        self.model.heads.MODE = 'FC'


        for epoch in range(EPOCH):
            loss_total = 0
            print(f'start training for epoch {epoch} of {EPOCH}')
            time_stamp_start = time.strftime("%H:%M:%S", time.localtime()) 
            for id,data in enumerate(self.train_set):
                if id>max_id:
                    break
                target = data['targets'].to(device)
                images = (data['images']/255.0).to(device)

                if self.cfg.ATTACKMETHOD == 'MUAP':
                    adv_images = attack_method(images,target)
                else:
                    adv_images = attack_method(images,target)

                if id % frequency==0:
                    print(f'ssim = {eval_ssim(images,adv_images)} in epoch {epoch} of {id}')
                optimizer.zero_grad()

                output_clean = self.model(make_dict(images,target))
                output_dirty = self.model(make_dict(adv_images,target))
                
                loss_clean_dict = self.model.losses(output_clean,target)
                loss_dirty_dict = self.model.losses(output_dirty,target)
                
                loss_clean = sum(loss_clean_dict.values())
                loss_dirty = sum(loss_dirty_dict.values())

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
        #mse = nn.MSELoss(reduction='sum')
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        def odfa(f1,f2):
            return loss_fn(-f1,f2)

        eps=0.05
        eps_iter=1.0/255.0

        dict = {
            'FGSM'    :FGSM  (self.temp_cfg,self.temp_model, loss_fn, eps=eps, targeted=False),
            'IFGSM'   :IFGSM (self.temp_cfg,self.temp_model, loss_fn, eps=eps, eps_iter=eps_iter,targeted=False,rand_init=False),
            'MIFGSM'  :MIFGSM(self.temp_cfg,self.temp_model, loss_fn, eps=eps, eps_iter=eps_iter,targeted=False,decay_factor=1),
            'ODFA'    :ODFA  (self.temp_cfg,self.temp_model, odfa,eps=eps, eps_iter=eps_iter,targeted=True,rand_init=False),
            'SSAE'    :self.SSAE_generator,
            'MISR'    :self.MISR_generator,
            'MUAP'    :MUAP(self.temp_cfg,self.temp_model)
        }
        return dict[self.cfg.ATTACKMETHOD]


    