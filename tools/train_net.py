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
from fastreid.utils.reid_patch import get_query_set,get_def_query_set, record
from fastreid.utils.attack_patch import attack
from fastreid.utils.defense_patch import defense
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

    #     Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

    #    res = DefaultTrainer.test(cfg, model)
    # #     return res

    # trainer = DefaultTrainer(cfg)

    # trainer.resume_or_load(resume=args.resume)
    # trainer.train()
    print('Start attack and defense.')
    query_set=get_query_set(cfg)

    if args.attack_by:
        print('start attack')
        query_cfg = DefaultTrainer.auto_scale_hyperparams(cfg,query_set.dataset.num_classes)
        model = DefaultTrainer.build_model(query_cfg)  #启用baseline
        Checkpointer(model).load(query_cfg.MODEL.WEIGHTS)  # load trained model
        attack(query_cfg,model,query_set)
        pure_result=DefaultTrainer.test(query_cfg, model)       #test用于测试原始的query与gallery合成的test_set
        adv_result =DefaultTrainer.advtest(query_cfg, model)     #advtest用于测试adv_query与gallery合成的test_set

    if args.defend_by:
        print('start defense')
        def_adv_result =defense(cfg,query_set)

    record(cfg,pure_result,adv_result,def_adv_result)
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
