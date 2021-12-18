

from fastreid.engine import DefaultTrainer
from fastreid.utils.attack_patch.attack_algorithm import *
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.reid_patch import CHW_to_HWC, change_preprocess_image, classify_test_set, get_query_set, get_result, save_image
import skimage
from ..attack_algorithm import *


class ClassificationAttack:
    '''
    use classification attack in Reid problem , only attack query set images
    
    '''
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.target = True if cfg.ATTACKTYPE=='T' else False
        self.model_path = self.cfg.MODEL.TESTSET_TRAINED_WEIGHT
        self.default_setup()


    def default_setup(self):
        self.query_data_loader = get_query_set(self.cfg)
        
        # train the classify layer for query set, and the model will be saved in 'self.model_path'
        classify_test_set(self.cfg,self.query_data_loader) 

        cfg = DefaultTrainer.auto_scale_hyperparams(self.cfg,self.query_data_loader.dataset.num_classes)
        self.model = DefaultTrainer.build_model_main(cfg)  # use baseline_train
        self.model.preprocess_image=change_preprocess_image(cfg) # re-range the input size to [0,1]
        Checkpointer(self.model).load(self.model_path)  # load trained model

        self.SSIM=0


    def attack_images(self,images):
        adversary=self.match_attack_method(self.cfg,self.model,self.target)
        images = adversary(images)  
        return images.clone().detach()

    def evaluate(self,images1,images2):
        size = images1.shape[0]
        SSIM = 0
        for i in range(size):
            image1 = CHW_to_HWC(images1[i].cpu())
            image2 = CHW_to_HWC(images2[i].cpu())
            SSIM += skimage.measure.compare_ssim(image1,image2,multichannel=True)
        SSIM/=size
        self.SSIM+=SSIM


    def get_result(self):
        self.SSIM/=len(self.query_data_loader)
        return get_result(self.cfg,self.cfg.MODEL.WEIGHTS,'attack'),self.SSIM

    def attack(self):
        for _ ,data in enumerate(self.query_data_loader):
            images = (data['images']/255)
            path = data['img_paths']
            adv_images = self.attack_images(images)
            self.evaluate(images,adv_images)

            save_image(adv_images,path,'adv_query')


    def match_attack_method(self,cfg,model,target):
        eps=0.05
        eps_iter=1.0/255.0
        dict = {
                'C-FGSM'  :FGSM(cfg, model, eps = eps,targeted = target),
                'C-IFGSM' :IFGSM(cfg,model, eps=eps, eps_iter=eps_iter, targeted=target),
                'C-MIFGSM':MIFGSM(cfg,model, eps=eps, eps_iter=eps_iter, targeted=target, decay_factor=1.),
                'CW'      :CW(cfg,model,confidence=0,max_iterations=5000),
                }
        return dict[cfg.ATTACKMETHOD]


