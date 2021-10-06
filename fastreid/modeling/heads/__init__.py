# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build import REID_HEADS_REGISTRY, build_heads,build_train_heads,build_feature_heads

# import all the meta_arch, so they will be registered
from .embedding_head import EmbeddingHead
from .training_head import TrainingHead
from .clas_head import ClasHead
from .feature_heads import FeatureHead
