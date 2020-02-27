#!/usr/bin/env python
"""
"""

from spiegel.estimator.tf_estimator_base import TFEstimatorBase
import tensorflow as tf
from tensorflow.keras import layers

class CNN(TFEstimatorBase):
    """
    :param input_shape: Shape of matrix that will be passed to model input
    :type input_shape: tuple
    :param num_outputs: Number of outputs the model has
    :type numOuputs: int
    :param kwargs: optional keyword arguments to pass to
        :class:`spiegel.estimator.TFEstimatorBase`
    """

    def __init__(self, input_shape, num_outputs, **kwargs):
        """
        Constructor
        """

        super().__init__(input_shape, num_outputs, **kwargs)


    def build_model(self):
        """
        Construct CNN Model
        """

        self.model = tf.keras.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), strides=2, dilation_rate=(1,1),
                                     input_shape=self.input_shape,
                                     activation='relu'))
        self.model.add(layers.Conv2D(64, (3, 3), strides=2, dilation_rate=(1,1),
                                     activation='relu'))
        self.model.add(layers.Conv2D(128, (3, 3), strides=2, dilation_rate=(1,1),
                                     activation='relu'))
        self.model.add(layers.Conv2D(256, (3, 3), strides=2, dilation_rate=(1,1),
                                     activation='relu'))
        self.model.add(layers.Conv2D(256, (2, 2), strides=2, dilation_rate=(1,1),
                                     activation='relu'))
        self.model.add(layers.Conv2D(512, (2, 2), strides=2, dilation_rate=(1,1),
                                     activation='relu'))
        self.model.add(layers.Dropout(0.25))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(self.num_outputs))

        self.model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=TFEstimatorBase.rms_error,
            metrics=['accuracy']
        )
