#!/usr/bin/env python

"""
Init file for estimators
"""

from .estimator_base import EstimatorBase
from .tf_estimator_base import TFEstimatorBase

from .basic_ga import BasicGA
from .nsga3 import NSGA3

from .conv6 import Conv6
from .highway_bi_lstm import HighwayBiLSTM
from .highway_layer import HighwayLayer
from .lstm import LSTM
from .mlp import MLP

from .tf_epoch_logger import TFEpochLogger
