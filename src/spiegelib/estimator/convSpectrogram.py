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

        model.add(layers.Conv2D(64, 5, 2, padding='same', input_shape=self.input_shape))
        assert (model.output_shape == (None, 64, 64, 64))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, 5, 2, padding='same'))
        assert (model.output_shape == (None, 32, 32, 128))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(256, 5, 2, padding='same'))
        assert (model.output_shape == (None, 16, 16, 256))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(512, 5, 2, padding='same'))
        assert (model.output_shape == (None, 8, 8, 512))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(1024, 5, 2, padding='same'))
        assert (model.output_shape == (None, 4, 4, 1024))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(self.num_outputs))

        self.model = model
        self.model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.keras.losses.MSE,
        )
