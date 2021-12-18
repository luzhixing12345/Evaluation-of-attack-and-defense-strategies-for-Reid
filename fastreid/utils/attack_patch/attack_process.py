
#这里引用了github的项目advtorch的库
#如想详细了解请阅读源代码
#https://github.com/BorealisAI/advertorch

from .attack_algorithm import *
from .attack_type import *

device = 'cuda'
attack_type = ['QA','GA']
# C_Attack_algorithm_library=["FGSM",'IFGSM','MIFGSM','CW','ODFA']  #针对分类问题的攻击算法库
# #DDNL2 SPA CW时间太长了！！！
# R_Attack_algorithm_library=['SMA','FNA','MIS-RANKING']
# Attack_algorithm_library=C_Attack_algorithm_library+R_Attack_algorithm_library

class attack:

    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self.AttackProcess = self.initalization()

    def initalization(self):
        if self.cfg.ATTACKTYPE=='QA':
            QA = QueryAttack(self.cfg)
            return QA
        else :
            GA = GalleryAttack(self.cfg)
            return GA
    def start_attack(self):
        self.AttackProcess.attack()

    def get_result(self):
        return self.AttackProcess.get_result()






