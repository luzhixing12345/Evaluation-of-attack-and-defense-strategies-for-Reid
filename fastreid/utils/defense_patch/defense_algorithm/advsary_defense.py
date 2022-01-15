
import torch
import torch.nn as nn
from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.reid_patch import change_preprocess_image, eval_ssim, get_result, get_train_set
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
        self.model.preprocess_image = change_preprocess_image(self.cfg)
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
        loss_fun = nn.CrossEntropyLoss()

        EPOCH = 3
        max_id = 4000
        alpha = 0.5 # mixing ratio between clean data and attack data,and it represents the ratio of clean data 

        self.model.train()
        self.model.heads.mode = 'C'
        for epoch in range(EPOCH):
            loss_total = 0
            for id,data in enumerate(self.train_set):
                if id>max_id:
                    break
                target = data['targets'].to(device)
                images = (data['images']/255.0).to(device)

                if self.cfg.ATTACKMETHOD == 'MUAP':
                    adv_images = attack_method(images,target)
                else:
                    adv_images = attack_method(images,target)

                # if id%200==0:
                #     print(f'ssim = {eval_ssim(images,adv_images)} in epoch {epoch} of {id}')
                optimizer.zero_grad()

                logits_clean = self.model(images)
                logits_att = self.model(adv_images)

                loss = loss_fun(logits_clean,target)*alpha+loss_fun(logits_att,target)*(1-alpha)
                loss_total+=loss.item()
                loss.backward()
                optimizer.step()
            print('total_loss = ',loss_total,epoch)

            
        print('finished adv_training !')
        print('--------------------------------')
        Checkpointer(self.model,'model').save(f'{self.cfg.DEFENSEMETHOD}_{self.cfg.ATTACKMETHOD}_{self.cfg.DATASETS.NAMES[0]}_{self.cfg.CFGTYPE}')
        print('Successfully saved the adv_trained model !')


    def getAttackMethod(self):
        #mse = nn.MSELoss(reduction='sum')
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        def odfa(f1,f2):
            return loss_fn(f1,-f2)

        eps=0.05
        eps_iter=1.0/255.0
        self.target = False
        self.model.heads.mode = 'C'

        dict = {
            'FGSM'    :FGSM  (self.cfg,self.model, loss_fn, eps=eps, targeted=self.target),
            'IFGSM'   :IFGSM (self.cfg,self.model, loss_fn, eps=eps, eps_iter=eps_iter,targeted=self.target,rand_init=False),
            'MIFGSM'  :MIFGSM(self.cfg,self.model, loss_fn, eps=eps, eps_iter=eps_iter,targeted=self.target,decay_factor=1),
            'ODFA'    :ODFA  (self.cfg,self.model, odfa,eps=eps, eps_iter=eps_iter,targeted=not self.target,rand_init=False),
            'SSAE'    :self.SSAE_generator,
            'MISR'    :self.MISR_generator,
            'MUAP'    :MUAP(self.cfg,self.model)
        }
        return dict[self.cfg.ATTACKMETHOD]


    