#!/usr/bin/env python
"""
LSTM Deep Learning Model

Based on model proposed by Yee-King et al. [1]_
"""

import tensorflow as tf
from tensorflow.keras import layers

from spiegelib.estimator.tf_estimator_base import TFEstimatorBase

class LSTM(TFEstimatorBase):
    """
    :param input_shape: Shape of matrix that will be passed to model input
    :type input_shape: tuple
    :param num_outputs: Number of outputs the model has
    :type numOuputs: int
    :param kwargs: optional keyword arguments to pass to
        :class:`spiegelib.estimator.TFEstimatorBase`
    """

    def __init__(self, input_shape, num_outputs, **kwargs):
        """
        Constructor
        """

        super().__init__(input_shape, num_outputs, **kwargs)


    def build_model(self, highway_layers=100):
        """
        Construct LSTM Model

        :param highway_layers: dimensionality of outer space of hidden layers,
            defaults to 100
        :type highway_layers: int, optional
        """

        self.model = tf.keras.Sequential()
        self.model.add(layers.LSTM(highway_layers, input_shape=self.input_shape,
                       return_sequences=True))
        self.model.add(layers.LSTM(highway_layers, return_sequences=True))
        self.model.add(layers.LSTM(highway_layers))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(
            self.num_outputs,
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            bias_initializer=tf.random_normal_initializer(stddev=0.01),
        ))

        self.model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=TFEstimatorBase.rms_error,
            metrics=['accuracy']
        )
