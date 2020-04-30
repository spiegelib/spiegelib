#!/usr/bin/env python
"""
Convolutional Neural Network based on the 6-layer deep model proposed by
Barkan et al. [1]_
"""

import tensorflow as tf
from tensorflow.keras import layers

from spiegelib.estimator.tf_estimator_base import TFEstimatorBase

class Conv6(TFEstimatorBase):
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

        self.model = tf.keras.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), strides=(2,2), dilation_rate=(1,1),
                                     input_shape=self.input_shape,
                                     activation='relu'))
        self.model.add(layers.Conv2D(71, (3, 3), strides=(2,2), dilation_rate=(1,1),
                                     activation='relu'))
        self.model.add(layers.Conv2D(128, (3, 4), strides=(2,3), dilation_rate=(1,1),
                                     activation='relu'))
        self.model.add(layers.Conv2D(128, (3, 3), strides=(2,2), dilation_rate=(1,1),
                                     activation='relu'))
        self.model.add(layers.Conv2D(128, (3, 3), strides=(2,2), dilation_rate=(1,1),
                                     activation='relu'))
        self.model.add(layers.Conv2D(128, (3, 3), strides=(1,2), dilation_rate=(1,1),
                                     activation='relu'))
        self.model.add(layers.Dropout(0.20))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(self.num_outputs))

        self.model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=TFEstimatorBase.rms_error,
            metrics=['accuracy']
        )
