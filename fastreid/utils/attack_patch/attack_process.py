
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

def attack(cfg):
    if cfg.ATTACK_C:
        CA = ClassificationAttack(cfg)
        CA.attack()
        return CA.get_result()
    
    else :
        if cfg.ATTACKTYPE=='QA':
            QA = QueryAttack(cfg)
            QA.attack()
            return QA.get_result()
        else :
            GA = GalleryAttack(cfg)
            GA.attack()
            return GA.get_result()






