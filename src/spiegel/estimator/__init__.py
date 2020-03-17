#!/usr/bin/env python

"""
Init file for estimators
"""

from .estimator_base import EstimatorBase
from .tf_estimator_base import TFEstimatorBase

# Genetic Algorithms
from .basic_ga import BasicGA
from .nsga3 import NSGA3

# Deep learning models
from .conv6 import Conv6
from .hwy_blstm import HwyBLSTM
from .lstm import LSTM
from .mlp import MLP

# Extra layers and utils for TF
from .highway_layer import HighwayLayer
from .tf_epoch_logger import TFEpochLogger
