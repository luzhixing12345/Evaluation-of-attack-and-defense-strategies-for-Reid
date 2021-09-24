# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build import META_ARCH_REGISTRY, build_model,build_model_for_pretrain,build_model_for_attack


# import all the meta_arch, so they will be registered
from .baseline import Baseline
from .baseline_train_query import Baseline_pretrain
from .baseline_attack import Baseline_attack
from .mgn import MGN
from .moco import MoCo
from .distiller import Distiller
