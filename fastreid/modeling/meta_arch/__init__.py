# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build import META_ARCH_REGISTRY, build_model


# import all the meta_arch, so they will be registered
from .baseline import Baseline
from .baseline_for_attack import Baseline_For_Attack
from .baseline_for_defense import Baseline_For_Defense
from .mgn import MGN
from .moco import MoCo
from .distiller import Distiller
