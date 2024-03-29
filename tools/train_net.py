#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import sys

sys.path.append('.')
from fastreid.config import get_cfg
from fastreid.engine import (DefaultTrainer, default_argument_parser,
                             default_setup, launch)
from fastreid.utils.attack_patch.attack_process import attack
from fastreid.utils.defense_patch.defense_process import defense
from fastreid.utils.reid_patch import (
                                       analyze_configCondition, easylog, get_result, get_train_set, move_model_pos, print_info,
                                       record)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    cfg = analyze_configCondition(args,cfg)
    cfg.freeze()
    default_setup(cfg, args)

    return cfg


def main(args):
    cfg = setup(args)
    
    if args.train:
        # only train the model and then return
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=args.resume)
        trainer.train()
        move_model_pos(cfg)
        return 

    # query_set = get_query_set(cfg)  
    # gallery_set = get_gallery_set(cfg)
    # train_set = get_train_set(cfg)  
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.WEIGHTS=f'./model/{cfg.DATASETS.NAMES[0]}_{cfg.CFGTYPE}.pth'

    # the final result conclude some evaluating indicator, which you can get from ./fastreid/utils/reid_patch.py
    # if you want to update or change it, you can find the computing method in fastreid\evaluation\reid_evaluation.py
    # and add your own computing method and indicator in it 

    # we will totally record four results as below
    pure_result=None
    att_result=None
    def_result=None
    def_adv_result=None
    
    # SSIM & def_SSIM are two evaluation indicators for image similarity
    SSIM=None
    def_SSIM = None
    
    # the model positions have been already defined in fastreid\config\defaults.py 
    # _C.MODEL.WEIGHTS                = "./model/model_final.pth"       the origin model after training 
    # _C.MODEL.TESTSET_TRAINED_WEIGHT = './model/test_trained.pth'      the model that can classify well in query set
    # _C.MODEL.DEFENSE_TRAINED_WEIGHT = "./model/def_trained.pth"       the model that can defense well towards the attack method


    # Dataset information :
    #
    # market1501  pic    target                  DukeMTMC   pic     target
    # query      3368    750                     query      2228    702
    # gallery   15913    751                     gallery    17661   1110
    # train     12936                            train      16522   702
    
    if not cfg.ONLYDEFENSE:
        # if only defense, we just test use the pretrained defense model testing in pure datasets
        pure_result = get_result(cfg,cfg.MODEL.WEIGHTS,'pure')
    
    if args.attack and not cfg.ONLYDEFENSE:
        # attack 
        print_info('start attack')
        AttackProcess = attack(cfg)
        AttackProcess.start_attack()
        att_result,SSIM = AttackProcess.get_result()
        print('ssim = ',SSIM)

    if args.defense:
        print_info('start defense')
        DefenseProcess = defense(cfg)
        DefenseProcess.start_defense()
        def_result,def_adv_result,def_SSIM= DefenseProcess.get_result()

    if args.record:
        record(cfg, pure_result, att_result, def_result, def_adv_result,SSIM,def_SSIM)
        print_info('the results were recorded in the excel in the root path')
    
    easylog(cfg, pure_result, att_result, def_result, def_adv_result,SSIM,def_SSIM)
    # if args.log: #default True
    #     record_order(cfg,pure_result_to_save,att_result_to_save,def_result_to_save,def_adv_result_to_save,save_pic = args.save_pic)
    

    print_info("You have finished!")



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
