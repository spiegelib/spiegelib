#!/usr/bin/env python
"""
Bidirection LSTM with Highway Layers - Deep Learning Model

Based on model proposed by Yee-King et al.

"Automatic programming of VST sound synthesizers using deep networks and other techniques."
Yee-King, Matthew John, Leon Fedden, and Mark d'Inverno.
IEEE Transactions on Emerging Topics in Computational Intelligence 2.2 (2018): 150-159.
"""

from spiegel.estimator.tf_estimator_base import TFEstimatorBase
import tensorflow as tf
from tensorflow.keras import layers
from spiegel.estimator.highway_layer import Highway

class HighwayBiLSTM(TFEstimatorBase):
    """
    :param inputShape: Shape of matrix that will be passed to model input
    :type inputShape: tuple
    :param numOutputs: Number of outputs the model has
    :type numOuputs: int
    :param kwargs: optional keyword arguments to pass to :class:`spiegel.estimator.TFEstimatorBase`
    """

    def __init__(self, inputShape, numOutputs, lstmSize=128, highwayLayers=6, **kwargs):
        """
        Constructor
        """

        self.lstmSize = lstmSize
        self.highwayLayers = highwayLayers
        super().__init__(inputShape, numOutputs, **kwargs)


    def buildModel(self):
        """
        Construct LSTM Model

        :param hiddenSize: dimensionality of outer space of hidden layers, defaults to 128
        :type hiddenSize: int, optional
        """

        self.model = tf.keras.Sequential()
        self.model.add(
            layers.Bidirectional(
                layers.LSTM(self.lstmSize),
                input_shape=self.inputShape,
                merge_mode='concat'
            )
        )
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(
            64,
            activation='elu',
            activity_regularizer=tf.keras.regularizers.l2()
        ))

        self.hwy = None

        # Add highway layers
        for i in range(self.highwayLayers):
            self.hwy = Highway(
                activation='elu',
                transform_dropout=0.2,
                activity_regularizer=tf.keras.regularizers.l2()
            )
            self.model.add(self.hwy)

        self.model.add(layers.Dense(
            self.numOutputs,
            activation='elu',
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            bias_initializer=tf.random_normal_initializer(stddev=0.01),
        ))

        self.model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=TFEstimatorBase.rootMeanSquaredError,
            metrics=['accuracy']
        )
