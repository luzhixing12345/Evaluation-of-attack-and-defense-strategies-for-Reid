


from .defense_algorithm import *
from ..attack_patch.attack_process import attack

# G_Defense_algorithm_library=['ADV','GRA','DIST']
# R_Defense_algorithm_library=['GOAT','EST','SES','PNP']
# Defense_algorithm_library=G_Defense_algorithm_library+R_Defense_algorithm_library

class defense:

    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self.initalization()
        self.UseAttack = self.cfg.ATTACKTYPE!=''
        self.pretrained = self.cfg.DEFENSEPRETRAINED

    def initalization(self):
        dict = {
            'ADV':adversary_defense,
            'GOAT':goat_defense,
            'EST':robrank_defense,
            'SES':robrank_defense,
            'PNP':robrank_defense,
        }
        self.DefenseProcess = dict[self.cfg.DEFENSEMETHOD](self.cfg)

    def start_defense(self):
        if self.pretrained:
            print('Use the pretrained defense model')
        else:
            self.DefenseProcess.defense()
        
        if self.UseAttack:
            self.cfg.defrost()
            self.cfg.MODEL.WEIGHTS = self.cfg.MODEL.DEFENSE_TRAINED_WEIGHT
            self.cfg.freeze()

            self.AttackProcess = attack(self.cfg)
            self.AttackProcess.start_attack()

    def get_result(self):
        def_result =  self.DefenseProcess.get_defense_result()
        def_att_result = None
        
        if self.UseAttack:
            def_att_result = self.AttackProcess.get_result()
        print(def_result)
        print(def_att_result)
        return def_result,def_att_result
        