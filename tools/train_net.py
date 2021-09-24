#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.reid_patch import evaluate_misMatch, get_query_set, record,train_query_set,get_pure_result
from fastreid.utils.attack_patch.attack_patch import attack_C,attack_R
from fastreid.utils.defense_patch.defense_patch import defense
import torch
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

    '''
        攻击思路：
                白盒攻击：通过训练集训练出model，得到logits值，针对请求集(query_set)的图像做出对应的攻击
                         并在注册集(gallery_set)上测试其reid能力
    '''
    # if args.eval_only:
    #     cfg.defrost()
    #     cfg.MODEL.BACKBONE.PRETRAIN = False
    #     model = DefaultTrainer.build_model(cfg)

    #Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

    #res = DefaultTrainer.test(cfg, model)
    # #     return res

    # trainer = DefaultTrainer(cfg)

    # trainer.resume_or_load(resume=args.resume)
    # trainer.train()
    query_set=get_query_set(cfg)
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    
    #保存的结果，默认值都为0
    pure_result={'Rank-1':0,'Rank-5':0,'Rank-10':0,'mAP':0,'mINP':0,'metric':0,'misMatch':0}
    adv_result={'Rank-1':0,'Rank-5':0,'Rank-10':0,'mAP':0,'mINP':0,'metric':0,'misMatch':0}
    def_result={'Rank-1':0,'Rank-5':0,'Rank-10':0,'mAP':0,'mINP':0,'metric':0,'misMatch':0}
    def_adv_result={'Rank-1':0,'Rank-5':0,'Rank-10':0,'mAP':0,'mINP':0,'metric':0,'misMatch':0}

    if args.query_train:
        cfg = train_query_set(cfg,query_set)
        #模型保存的位置在(./model/query_trained.pth)
        print('finished the train for query set ')
        print('---------------------------------------------')
    
    #注:已经预先在fastreid\config\defaults.py中定义了
    # _C.MODEL.QUERYSET_TRAINED_WEIGHT = './model/query_trained.pth'
    # _C.MODEL.TRAINSET_TRAINED_WEIGHT = "./model/pretrained.pth"
    # _C.MODEL.DEFENSE_TRAINED_WEIGHT = "./model/adv_trained.pth"
    #三个位置为模型保存的位置

    pure_result,std_model=get_pure_result(cfg,query_set)

    if args.attack:
        print('start attack')
        if args.C:#针对分类问题的攻击
            query_cfg = DefaultTrainer.auto_scale_hyperparams(cfg,query_set.dataset.num_classes)
            model = DefaultTrainer.build_model_for_attack(query_cfg)  #启用baseline_for_query_train
            Checkpointer(model).load(query_cfg.MODEL.QUERYSET_TRAINED_WEIGHT)  # load trained model
            
            attack_C(query_cfg,model,query_set)

            adv_result =DefaultTrainer.advtest(query_cfg, std_model)     #advtest用于测试adv_query与gallery合成的test_set
            pure_misMatch = evaluate_misMatch(std_model,query_set)
            pure_result['misMatch']=pure_misMatch
            adv_misMatch =evaluate_misMatch(model,query_set)
            adv_result['misMatch']=adv_misMatch
        elif args.R:#针对排序问题的攻击
            model = DefaultTrainer.build_model_for_attack(cfg)  #启用baseline_for_query_train
            Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
            adv_result=attack_R(cfg,model,query_set)
        else:
            print('You must directly claim which type of the attack method ,please check in the USE.md to change your arguments')
            raise
    if args.defense:
        #print('start defense')
        def_result,def_adv_result =defense(cfg,query_set)


    record(cfg,pure_result,adv_result,def_result,def_adv_result)
    print("You have finished the task !")


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
