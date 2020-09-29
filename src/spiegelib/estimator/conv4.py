#!/usr/bin/env python
"""
Convolutional Neural Network based on the 4-layer deep model proposed by
Barkan et al. [1]_
"""

import tensorflow as tf
from tensorflow.keras import layers

from spiegelib.estimator.tf_estimator_base import TFEstimatorBase


class Conv4(TFEstimatorBase):
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

    def build_model(self):
        """
        Construct 4-layer CNN Model
        """

        self.model = tf.keras.Sequential()
        self.model.add(layers.Conv2D(32, (3, 4), strides=(2, 3), input_shape=self.input_shape, activation='relu', padding='same'))
        self.model.add(layers.Conv2D(65, (3, 4), strides=(2, 3), activation='relu', padding='same'))
        self.model.add(layers.Conv2D(105, (3, 4), strides=(2, 3), activation='relu', padding='same'))
        self.model.add(layers.Conv2D(128, (4, 5), strides=(3, 4), activation='relu', padding='same'))
        self.model.add(layers.Dropout(0.30))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(512))
        self.model.add(layers.Dense(self.num_outputs))

        self.model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=TFEstimatorBase.rms_error,
            metrics=['accuracy']
        )
