

import torch
from fastreid.engine import DefaultTrainer
from fastreid.utils.attack_patch.attack_algorithm import *
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.reid_patch import change_preprocess_image, classify_test_set, evaluate_ssim, get_query_set, get_result, save_image

from ..attack_algorithm import *
from fastreid.utils.advertorch.attacks import MomentumIterativeAttack,LinfPGDAttack,CarliniWagnerL2Attack

class ClassificationAttack:
    '''
    use classification attack in Reid problem , only attack query set images
    
    '''
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.target = True if cfg.ATTACKTYPE=='T' else False
        self.model_path = "./model/test_trained.pth"
        self.default_setup()


    def default_setup(self):
        self.query_data_loader = get_query_set(self.cfg)
        
        # train the classify layer for query set, and the model will be saved in 'self.model_path'
        classify_test_set(self.cfg,self.query_data_loader) 

        cfg = DefaultTrainer.auto_scale_hyperparams(self.cfg,self.query_data_loader.dataset.num_classes)
        self.model = DefaultTrainer.build_model_main(cfg)  # use baseline_train
        self.model.preprocess_image=change_preprocess_image(cfg) # re-range the input size to [0,1]
        Checkpointer(self.model).load(self.model_path)  # load trained model


    def attack_images(self,images):
        adversary=self.match_attack_method(self.cfg,self.model,self.target)
        images = adversary(images)  
        return images.clone().detach()

    def get_result(self):
        return get_result(self.cfg,self.cfg.MODEL.WEIGHTS,'attack'),evaluate_ssim(self.cfg)

    def attack(self):
        for _ ,data in enumerate(self.query_data_loader):
            images = (data['images']/255)
            path = data['img_paths']
            images = self.attack_images(images)

            save_image(images,path,'adv_query')


    def match_attack_method(self,cfg,model,target):
        eps=0.05
        eps_iter=1.0/255.0
        dict = {
                'C-FGSM'  :FGSM(cfg, model, eps = eps,targeted = target),
                'C-IFGSM' :IFGSM(cfg,model, eps=eps, eps_iter=eps_iter, targeted=target),
                'C-MIFGSM':MIFGSM(cfg,model, eps=eps, eps_iter=eps_iter, targeted=target, decay_factor=1.),
                'CW'      :CW(cfg,model),
                }
        return dict[cfg.ATTACKMETHOD]


