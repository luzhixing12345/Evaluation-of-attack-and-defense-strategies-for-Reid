

import copy

from numpy import True_

from fastreid.utils.reid_patch import get_result
from .defense_algorithm import *
from ..attack_patch.attack_process import attack
from tabulate import tabulate
from termcolor import colored
# G_Defense_algorithm_library=['ADV','GRA','DIST']
# R_Defense_algorithm_library=['GOAT','EST','SES','PNP']
# Defense_algorithm_library=G_Defense_algorithm_library+R_Defense_algorithm_library

class defense:

    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self.initalization()
        self.QA_attack = self.cfg.ATTACKTYPE=='QA'
        self.GA_attack = self.cfg.ATTACKTYPE=='GA'
        self.pretrained = self.cfg.DEFENSEPRETRAINED
        
    def initalization(self):
        dict = {
            'ADV':adversary_defense,
            'GOAT':goat_defense,
            'EST':robrank_defense,
            'SES':robrank_defense,
            'GRA':gradient_regulation_defense,
            'DIS':distillation_defense
        }
        if self.cfg.DEFENSEMETHOD == 'ADV':
            self.cfg.MODEL.DEFENSE_TRAINED_WEIGHT = f'./model/{self.cfg.DEFENSEMETHOD}_{self.cfg.ATTACKMETHOD}_{self.cfg.DATASETS.NAMES[0]}_{self.cfg.CFGTYPE}.pth'
        else:
            self.cfg.MODEL.DEFENSE_TRAINED_WEIGHT = f'./model/{self.cfg.DEFENSEMETHOD}_{self.cfg.DATASETS.NAMES[0]}_{self.cfg.CFGTYPE}.pth'
        temp_cfg = copy.deepcopy(self.cfg)
        temp_cfg.ATTACKTYPE = 'QA'
        self.DefenseProcess = dict[temp_cfg.DEFENSEMETHOD](temp_cfg)

    def start_defense(self):
        if self.pretrained:
            print('Use the pretrained defense model')
        else:
            self.DefenseProcess.defense()
        
        # if self.QA_attack:
        #     self.cfg.defrost()
        #     self.cfg.MODEL.WEIGHTS = self.cfg.MODEL.DEFENSE_TRAINED_WEIGHT
        #     self.cfg.freeze()

        #     self.AttackProcess = attack(self.cfg)
        #     self.AttackProcess.start_attack()

    def get_result(self):
        def_result =  self.DefenseProcess.get_defense_result()
        self.print_csv_format(def_result)
        def_att_result = None
        def_SSIM = 0
           
        if self.QA_attack:
            def_att_result = get_result(self.cfg,self.cfg.MODEL.DEFENSE_TRAINED_WEIGHT,'attack')
        if self.GA_attack:
            self.AttackProcess = attack(self.cfg)
            self.AttackProcess.GA_preprocess()
            self.AttackProcess.AttackProcess.attack(defense_type=True)
            def_att_result,def_SSIM = self.AttackProcess.get_result()
        # if self.QA_attack:
        #     def_att_result,def_SSIM= self.AttackProcess.get_result()
        #     self.print_csv_format(def_att_result)
        
            # print('def-SSIM = ',def_SSIM)
        return def_result,def_att_result,def_SSIM

    def print_csv_format(self,results):
        """
        Print main metrics in a format similar to Detectron2,
        so that they are easy to copypaste into a spreadsheet.
        Args:
            results (OrderedDict): {metric -> score}
        """
        # unordered results cannot be properly printed
        dataset_name = self.cfg.DATASETS.NAMES[0]
        metrics = ["Dataset"] + [k for k in results]
        csv_results = [(dataset_name, *list(results.values()))]

        # tabulate it
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            floatfmt=".2f",
            headers=metrics,
            numalign="left",
        )
        print(colored(table, "cyan"))
        