#!/usr/bin/env python
"""
Deep CNN for estimating parameters from spectrograms
"""

import tensorflow as tf
from tensorflow.keras import layers

from spiegelib.estimator.tf_estimator_base import TFEstimatorBase


class ConvSpectrogram(TFEstimatorBase):
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
        Construct 6-layer CNN Model
        """

        model = tf.keras.Sequential()

        model.add(layers.Conv2D(32, 3, 2, padding='same', input_shape=self.input_shape))
        assert (model.output_shape == (None, 64, 64, 32))
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(64, 3, 2, padding='same'))
        assert (model.output_shape == (None, 32, 32, 64))
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(128, 3, 2, padding='same'))
        assert (model.output_shape == (None, 16, 16, 128))
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(128, 3, 2, padding='same'))
        assert (model.output_shape == (None, 8, 8, 128))
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(128, 3, 1, padding='same'))
        assert (model.output_shape == (None, 8, 8, 128))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.4))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(self.num_outputs, activation='sigmoid'))

        self.model = model
        self.model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=TFEstimatorBase.rms_error,
            metrics=['accuracy']
        )
