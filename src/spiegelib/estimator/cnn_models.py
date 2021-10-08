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

    def __init__(
        self,
        input_shape,
        num_outputs,
        dropout=None,
        batch_norm=False,
        fc_layers=1,
        fc_layer_size=512,
        **kwargs
    ):
        """
        Constructor
        """

        self.dropout = dropout
        self.batch_norm = batch_norm
        self.fc_layers = fc_layers
        self.fc_layer_size = fc_layer_size
        super().__init__(input_shape, num_outputs, **kwargs)

    def build_model(self):
        """
        Construct 6-layer CNN Model
        """

        self.model = tf.keras.Sequential()
        self.model.add(
            layers.Conv2D(
                32,
                (3, 3),
                strides=(2, 2),
                input_shape=self.input_shape,
                activation="relu",
                padding="same",
            )
        )
        if self.batch_norm:
            self.model.add(layers.BatchNormalization())

        self.model.add(
            layers.Conv2D(71, (3, 3), strides=(2, 2), activation="relu", padding="same")
        )
        if self.batch_norm:
            self.model.add(layers.BatchNormalization())

        self.model.add(
            layers.Conv2D(
                128, (3, 4), strides=(2, 3), activation="relu", padding="same"
            )
        )
        if self.batch_norm:
            self.model.add(layers.BatchNormalization())

        self.model.add(
            layers.Conv2D(
                128, (3, 3), strides=(2, 2), activation="relu", padding="same"
            )
        )
        if self.batch_norm:
            self.model.add(layers.BatchNormalization())

        self.model.add(
            layers.Conv2D(
                128, (3, 3), strides=(2, 2), activation="relu", padding="same"
            )
        )
        if self.batch_norm:
            self.model.add(layers.BatchNormalization())

        self.model.add(
            layers.Conv2D(
                128, (3, 3), strides=(1, 2), activation="relu", padding="same"
            )
        )
        if self.batch_norm:
            self.model.add(layers.BatchNormalization())

        if self.dropout is not None:
            self.model.add(layers.Dropout(self.dropout))

        self.model.add(layers.Flatten())

        for i in range(self.fc_layers):
            self.model.add(layers.Dense(self.fc_layer_size, activation="relu"))
            if self.dropout is not None:
                self.model.add(layers.Dropout(self.dropout))

        self.model.add(layers.Dense(self.num_outputs))

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
        )


class Conv6Small(TFEstimatorBase):
    """
    :param input_shape: Shape of matrix that will be passed to model input
    :type input_shape: tuple
    :param num_outputs: Number of outputs the model has
    :type numOuputs: int
    :param kwargs: optional keyword arguments to pass to
        :class:`spiegelib.estimator.TFEstimatorBase`
    """

    def __init__(
        self,
        input_shape,
        num_outputs,
        dropout=None,
        batch_norm=False,
        fc_layers=1,
        fc_layer_size=128,
        **kwargs
    ):
        """
        Constructor
        """

        self.dropout = dropout
        self.batch_norm = batch_norm
        self.fc_layers = fc_layers
        self.fc_layer_size = fc_layer_size
        super().__init__(input_shape, num_outputs, **kwargs)

    def build_model(self):
        """
        Construct 6-layer CNN Model
        """

        self.model = tf.keras.Sequential()
        self.model.add(
            layers.Conv2D(
                32,
                (3, 3),
                strides=(2, 2),
                input_shape=self.input_shape,
                activation="relu",
                padding="same",
            )
        )
        if self.batch_norm:
            self.model.add(layers.BatchNormalization())

        self.model.add(
            layers.Conv2D(32, (3, 3), strides=(2, 2), activation="relu", padding="same")
        )
        if self.batch_norm:
            self.model.add(layers.BatchNormalization())

        self.model.add(
            layers.Conv2D(64, (3, 3), strides=(2, 2), activation="relu", padding="same")
        )
        if self.batch_norm:
            self.model.add(layers.BatchNormalization())

        self.model.add(
            layers.Conv2D(64, (3, 3), strides=(2, 2), activation="relu", padding="same")
        )
        if self.batch_norm:
            self.model.add(layers.BatchNormalization())

        self.model.add(
            layers.Conv2D(
                128, (3, 3), strides=(2, 2), activation="relu", padding="same"
            )
        )
        if self.batch_norm:
            self.model.add(layers.BatchNormalization())

        self.model.add(
            layers.Conv2D(
                128, (3, 3), strides=(2, 2), activation="relu", padding="same"
            )
        )
        if self.batch_norm:
            self.model.add(layers.BatchNormalization())

        if self.dropout is not None:
            self.model.add(layers.Dropout(self.dropout))

        self.model.add(layers.Flatten())

        for i in range(self.fc_layers):
            self.model.add(layers.Dense(self.fc_layer_size, activation="relu"))
            if self.dropout is not None:
                self.model.add(layers.Dropout(self.dropout))

        self.model.add(layers.Dense(self.num_outputs))

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
        )


class Conv5(TFEstimatorBase):
    """
    :param input_shape: Shape of matrix that will be passed to model input
    :type input_shape: tuple
    :param num_outputs: Number of outputs the model has
    :type numOuputs: int
    :param kwargs: optional keyword arguments to pass to
        :class:`spiegelib.estimator.TFEstimatorBase`
    """

    def __init__(
        self,
        input_shape,
        num_outputs,
        dropout=None,
        batch_norm=False,
        fc_layers=1,
        fc_layer_size=512,
        **kwargs
    ):
        """
        Constructor
        """

        self.dropout = dropout
        self.batch_norm = batch_norm
        self.fc_layers = fc_layers
        self.fc_layer_size = fc_layer_size
        super().__init__(input_shape, num_outputs, **kwargs)

    def build_model(self):
        """
        Construct 5-layer CNN Model
        """

        self.model = tf.keras.Sequential()
        self.model.add(
            layers.Conv2D(
                32,
                (3, 3),
                strides=(2, 2),
                input_shape=self.input_shape,
                activation="relu",
                padding="same",
            )
        )
        if self.batch_norm:
            self.model.add(layers.BatchNormalization())

        self.model.add(
            layers.Conv2D(98, (3, 3), strides=(2, 2), activation="relu", padding="same")
        )
        if self.batch_norm:
            self.model.add(layers.BatchNormalization())

        self.model.add(
            layers.Conv2D(
                128, (3, 4), strides=(2, 3), activation="relu", padding="same"
            )
        )
        if self.batch_norm:
            self.model.add(layers.BatchNormalization())

        self.model.add(
            layers.Conv2D(
                128, (3, 5), strides=(2, 4), activation="relu", padding="same"
            )
        )
        if self.batch_norm:
            self.model.add(layers.BatchNormalization())

        self.model.add(
            layers.Conv2D(
                128, (3, 3), strides=(2, 2), activation="relu", padding="same"
            )
        )
        if self.batch_norm:
            self.model.add(layers.BatchNormalization())

        if self.dropout is not None:
            self.model.add(layers.Dropout(self.dropout))

        self.model.add(layers.Flatten())

        for i in range(self.fc_layers):
            self.model.add(layers.Dense(self.fc_layer_size, activation="relu"))
            if self.dropout is not None:
                self.model.add(layers.Dropout(self.dropout))

        self.model.add(layers.Dense(self.num_outputs))

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
        )


class Conv5Small(TFEstimatorBase):
    """
    :param input_shape: Shape of matrix that will be passed to model input
    :type input_shape: tuple
    :param num_outputs: Number of outputs the model has
    :type num_outputs: int
    :param kwargs: optional keyword arguments to pass to
        :class:`spiegelib.estimator.TFEstimatorBase`
    """

    def __init__(
        self,
        input_shape,
        num_outputs,
        dropout=None,
        batch_norm=False,
        fc_layers=1,
        fc_layer_size=128,
        **kwargs
    ):
        """
        Constructor
        """

        self.dropout = dropout
        self.batch_norm = batch_norm
        self.fc_layers = fc_layers
        self.fc_layer_size = fc_layer_size
        super().__init__(input_shape, num_outputs, **kwargs)

    def build_model(self):
        """
        Construct 5-layer smaller CNN Model
        """

        self.model = tf.keras.Sequential()
        self.model.add(
            layers.Conv2D(
                16,
                3,
                strides=(2, 2),
                input_shape=self.input_shape,
                activation="relu",
                padding="same",
            )
        )
        if self.batch_norm:
            self.model.add(layers.BatchNormalization())

        self.model.add(
            layers.Conv2D(32, 3, strides=(2, 2), activation="relu", padding="same")
        )
        if self.batch_norm:
            self.model.add(layers.BatchNormalization())

        self.model.add(
            layers.Conv2D(32, 3, strides=(2, 2), activation="relu", padding="same")
        )
        if self.batch_norm:
            self.model.add(layers.BatchNormalization())

        self.model.add(
            layers.Conv2D(64, 3, strides=(2, 2), activation="relu", padding="same")
        )
        if self.batch_norm:
            self.model.add(layers.BatchNormalization())

        self.model.add(
            layers.Conv2D(64, 3, strides=(2, 2), activation="relu", padding="same")
        )
        if self.batch_norm:
            self.model.add(layers.BatchNormalization())

        if self.dropout is not None:
            self.model.add(layers.Dropout(self.dropout))

        self.model.add(layers.Flatten())

        for i in range(self.fc_layers):
            self.model.add(layers.Dense(self.fc_layer_size, activation="relu"))
            if self.dropout is not None:
                self.model.add(layers.Dropout(self.dropout))

        self.model.add(layers.Dense(self.num_outputs))

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
        )
