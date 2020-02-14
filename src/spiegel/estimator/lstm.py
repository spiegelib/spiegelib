#!/usr/bin/env python
"""
LSTM Deep Learning Model

Based on model proposed by Yee-King et al.

"Automatic programming of VST sound synthesizers using deep networks and other techniques."
Yee-King, Matthew John, Leon Fedden, and Mark d'Inverno.
IEEE Transactions on Emerging Topics in Computational Intelligence 2.2 (2018): 150-159.
"""

from spiegel.estimator.tf_estimator_base import TFEstimatorBase
import tensorflow as tf
from tensorflow.keras import layers

class LSTM(TFEstimatorBase):
    """
    :param inputShape: Shape of matrix that will be passed to model input
    :type inputShape: tuple
    :param numOutputs: Number of outputs the model has
    :type numOuputs: int
    :param kwargs: optional keyword arguments to pass to :class:`spiegel.estimator.TFEstimatorBase`
    """

    def __init__(self, inputShape, numOutputs, **kwargs):
        """
        Constructor
        """

        super().__init__(inputShape, numOutputs, **kwargs)


    def buildModel(self, hiddenSize=100):
        """
        Construct LSTM Model

        :param hiddenSize: dimensionality of outer space of hidden layers, defaults to 100
        :type hiddenSize: int, optional
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
