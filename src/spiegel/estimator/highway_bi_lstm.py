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
from spiegel.estimator.highway_layer import HighwayLayer

class HighwayBiLSTM(TFEstimatorBase):
    """
    :param input_shape: Shape of matrix that will be passed to model input
    :type input_shape: tuple
    :param num_outputs: Number of outputs the model has
    :type num_outputs: int
    :param kwargs: optional keyword arguments to pass to
        :class:`spiegel.estimator.TFEstimatorBase`
    """

    def __init__(self, input_shape, num_outputs, lstm_size=128,
                 highway_layers=6, **kwargs):
        """
        Constructor
        """

        self.lstm_size = lstm_size
        self.highway_layers = highway_layers
        super().__init__(input_shape, num_outputs, **kwargs)


    def build_model(self):
        """
        Construct LSTM Model
        """

        self.model = tf.keras.Sequential()
        self.model.add(
            layers.Bidirectional(
                layers.LSTM(self.lstm_size),
                input_shape=self.input_shape,
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
        for i in range(self.highway_layers):
            self.hwy = HighwayLayer(
                activation='elu',
                transform_dropout=0.2,
                activity_regularizer=tf.keras.regularizers.l2()
            )
            self.model.add(self.hwy)

        self.model.add(layers.Dense(
            self.num_outputs,
            activation='elu',
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            bias_initializer=tf.random_normal_initializer(stddev=0.01),
        ))

        self.model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=TFEstimatorBase.rms_error,
            metrics=['accuracy']
        )
