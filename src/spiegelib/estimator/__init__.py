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
from .conv4 import Conv4
from .conv5 import Conv5
from .conv5_small import Conv5Small
from .conv6 import Conv6
from .conv6_small import Conv6Small
from .convSpectrogram import ConvSpectrogram
from .hwy_blstm import HwyBLSTM
from .lstm import LSTM
from .mlp import MLP

# Extra layers and utils for TF
from .highway_layer import HighwayLayer
from .tf_epoch_logger import TFEpochLogger
