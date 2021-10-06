# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build import META_ARCH_REGISTRY, build_model,build_model_main


# import all the meta_arch, so they will be registered
from .baseline import Baseline
from .baseline_train import Baseline_train
from .mgn import MGN
from .moco import MoCo
from .distiller import Distiller
