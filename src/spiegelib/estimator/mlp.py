#!/usr/bin/env python
"""
Simple Multi-layer Perceptron Deep Learning Model

Based on model proposed by Yee-King et al. [1]_
"""

import tensorflow as tf
from tensorflow.keras import layers

from spiegelib.estimator.tf_estimator_base import TFEstimatorBase

class MLP(TFEstimatorBase):
    """
    :param input_shape: Shape of matrix that will be passed to model input
    :type input_shape: tuple
    :param num_outputs: Number of outputs the model has
    :type numOuputs: int
    :param kwargs: optional keyword arguments to pass to :class:`spiegelib.estimator.TFEstimatorBase`
    """

    def __init__(self, input_shape, num_outputs, **kwargs):
        """
        Constructor
        """

        super().__init__(input_shape, num_outputs, **kwargs)


    def build_model(self):
        """
        Construct MLP Model
        """

        self.model = tf.keras.Sequential()
        self.model.add(layers.Dense(50, input_shape=self.input_shape, activation='relu'))
        self.model.add(layers.Dense(40, activation='relu'))
        self.model.add(layers.Dense(30, activation='relu'))
        self.model.add(layers.Dense(self.num_outputs))

        self.model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=TFEstimatorBase.rms_error,
            metrics=['accuracy']
        )
