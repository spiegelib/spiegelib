#!/usr/bin/env python
"""
Simple Multi-layer Perceptron Deep Learning Model

Based on model proposed by Yee-King et al. [1]_
"""
from typing import List

import tensorflow as tf
from tensorflow.keras import layers

from spiegelib.estimator.tf_estimator_base import TFEstimatorBase


class MLP(TFEstimatorBase):
    """
    :param input_shape: Shape of matrix that will be passed to model input
    :type input_shape: tuple
    :param num_outputs: Number of outputs the model has
    :type numOuputs: int
    :param kwargs: optional keyword arguments to pass to :class:`spiegelib.estimator.TFEstimatorBase`
    """

    def __init__(
        self,
        input_shape,
        num_outputs,
        hidden: List[int] = None,
        dropout: int = None,
        **kwargs
    ):
        """
        Constructor
        """

        if hidden is None:
            self.hidden = [50, 40, 30]
        elif isinstance(hidden, list):
            self.hidden = hidden
        else:
            raise TypeError("hidden must be a list of ints")

        self.dropout = dropout
        super().__init__(input_shape, num_outputs, **kwargs)

    def build_model(self):
        """
        Construct MLP Model
        """

        self.model = tf.keras.Sequential()

        # Add first hidden layer connected to the input
        self.model.add(
            layers.Dense(
                self.hidden[0], input_shape=self.input_shape, activation="relu"
            )
        )

        # Remaining hidden layers
        for hidden_size in self.hidden[1:]:
            self.model.add(layers.Dense(hidden_size, activation="relu"))

        if self.dropout is not None:
            self.model.add(layers.Dropout(self.dropout))

        self.model.add(layers.Dense(self.num_outputs))

        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
        )
