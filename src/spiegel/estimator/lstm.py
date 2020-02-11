#!/usr/bin/env python
"""
LSTM Deep Learning Model

Based on model proposed by Yee-King et al.
"""

from spiegel.estimator.tf_estimator_base import TFEstimatorBase
import tensorflow as tf
from tensorflow.keras import layers

class LSTM(TFEstimatorBase):

    def __init__(self, inputShape, outputShape):
        """
        Constructor
        """

        super().__init__(inputShape, outputShape)


    def buildModel(self, hiddenSize=100):
        """
        Construct LSTM Model
        """

        self.model = tf.keras.Sequential()
        self.model.add(layers.LSTM(hiddenSize, input_shape=self.inputShape, return_sequences=True))
        self.model.add(layers.LSTM(hiddenSize, return_sequences=True))
        self.model.add(layers.LSTM(hiddenSize))
        self.model.add(layers.Dense(
            self.numOutputs,
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            bias_initializer=tf.random_normal_initializer(stddev=0.01)
        ))

        self.model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=TFEstimatorBase.rootMeanSquaredError,
            metrics=['accuracy']
        )
