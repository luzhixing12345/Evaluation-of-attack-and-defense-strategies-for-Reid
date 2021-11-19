#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import sys

sys.path.append('.')
from fastreid.utils.reid_patch import evaluate_ssim, get_gallery_set, get_query_set, get_train_set, record,get_result, record_order, release_cuda_memory
from fastreid.utils.attack_patch.attack_patch import attack
from fastreid.utils.defense_patch.defense_patch import defense
from fastreid.utils.reid_patch import match_type
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.config import get_cfg





def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    
    if args.T:
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=args.resume)
        trainer.train()
        return 

    query_set = get_query_set(cfg)  
    gallery_set = get_gallery_set(cfg)
    train_set = get_train_set(cfg)  
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.WEIGHTS='./model/model_.pth'
    cfg.MODEL.WEIGHTS=cfg.MODEL.WEIGHTS[0:-4]+cfg.DATASETS.NAMES[0]+cfg.MODEL.WEIGHTS[-4:]


    # the final result conclude some evaluating indicator, which you can get from ./fastreid/utils/reid_patch.py
    # if you want to update or change it, you can find the computing method in fastreid\evaluation\reid_evaluation.py
    # and add your own computing method and indicator in it 

    # we will totally record four results as below
    pure_result=pure_result_to_save=None
    att_result=att_result_to_save=None
    def_result=def_result_to_save=None
    def_adv_result=def_adv_result_to_save=None  
    # the model positions have been already defined in fastreid\config\defaults.py 
    # _C.MODEL.WEIGHTS                = "./model/model_final.pth"       the origin model after training 
    # _C.MODEL.TESTSET_TRAINED_WEIGHT = './model/test_trained.pth'      the model that can classify well in query set
    # _C.MODEL.DEFENSE_TRAINED_WEIGHT = "./model/def_trained.pth"       the model that can defense well towards the attack method


    # market1501  pic    target                  DukeMTMC   pic     target
    # query      3368    750                     query      2228    702
    # gallery   15913    751                     gallery    17661   1110
    pure_result ,pure_result_to_save= get_result(cfg,cfg.MODEL.WEIGHTS,'pure')

    if args.attack:
        print('start attack')
        att_result,att_result_to_save,SSIM= attack(cfg,query_set,gallery_set,match_type(cfg,'attack'),pos='adv_query')
        print('ssim = ',SSIM)

    if args.defense:
        print('start defense')
        def_result,def_result_to_save, def_adv_result,def_adv_result_to_save = defense(cfg,train_set,query_set,gallery_set,match_type(cfg,'defense'))

    if args.record:
        record(cfg, pure_result, att_result, def_result, def_adv_result,SSIM)
        print('the results were recorded in the excel in the root path')
    record_order(cfg,pure_result_to_save,att_result_to_save,def_result_to_save,def_adv_result_to_save)
    
    
    print("You have finished all the jobs !")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
