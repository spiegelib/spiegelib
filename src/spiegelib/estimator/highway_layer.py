#!/usr/bin/env python
"""
Custom highway layer implemented in Keras.

Based on implementation by Kadam Parikh: https://github.com/ParikhKadam/Highway-Layer-Keras
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant


class HighwayLayer(layers.Layer):
    """
    Args:
        activation (optional): Activation function of name of built-in activation function.
        transform_gate_bias (optional): Initializer for transform gate bias vector
        transform_dropout (float, optional): Dropout rate between 0 and 1 for transform gate.
            Defaults to None and no dropout is applied.
        activity_regularizer: Optional activity regulizer applied to the regular dense layer
        kwargs: Optional keyword arguments passed to ``tf.keras.layers.Layer``. See
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer>`_.
    """

    def __init__(
        self,
        activation='relu',
        transform_gate_bias=-1,
        transform_dropout=None,
        activity_regularizer=None,
        **kwargs
    ):
        """
        Constructor
        """
        self.activation = activation
        self.transform_gate_bias = transform_gate_bias
        self.transform_dropout = transform_dropout
        self.activity_regularizer = activity_regularizer
        super(HighwayLayer, self).__init__(**kwargs)


    def build(self, input_shape):

        # Create a trainable weight variable for this layer.
        dim = input_shape[-1]
        transform_gate_bias_initializer = Constant(self.transform_gate_bias)
        input_shape_dense_1 = input_shape[-1]

        # Transform Gate
        self.dense_1 = layers.Dense(
            name='Dense_1',
            units=dim,
            bias_initializer=transform_gate_bias_initializer
        )

        # Regular Dense Layer
        self.dense_2 = layers.Dense(
            name='Dense_2',
            units=dim,
            activity_regularizer=self.activity_regularizer
        )

        super(HighwayLayer, self).build(input_shape)

    def call(self, x):
        dim = K.int_shape(x)[-1]

        # Transform gate operation
        transform_gate = self.dense_1(x)
        transform_gate = layers.Activation("sigmoid")(transform_gate)
        if self.transform_dropout:
            transform_gate = layers.Dropout(self.transform_dropout)(transform_gate)

        # Carry gate operation - determine how much to feedforward
        carry_gate = layers.Lambda(lambda x: 1.0 - x, output_shape=(dim,))(transform_gate)

        transformed_data = self.dense_2(x)
        transformed_data = layers.Activation(self.activation)(transformed_data)
        transformed_gated = layers.Multiply()([transform_gate, transformed_data])

        identity_gated = layers.Multiply()([carry_gate, x])
        value = layers.Add()([transformed_gated, identity_gated])
        return value

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config['activation'] = self.activation
        config['transform_gate_bias'] = self.transform_gate_bias
        return config
