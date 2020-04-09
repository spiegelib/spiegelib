#!/usr/bin/env python
"""
Abstract Base Class for Estimating Synthesizer Parameters using TensorFlow.

Wraps TensorFlow library calls and provides functionality for training and making
synthesizer parameter predictions. Inheriting classes must implement the ``build_model()``
method, which is automatically called upon construction and sets up the neural network
model.

Examples
^^^^^^^^

Here is an example of extending ``TFEstimatorBase`` to create a simple multi-layer
perceptron network::

        import tensorflow as tf
        from tensorflow.keras import layers
        import spiegelib.estimators import TFEstimatorBase

        class MLP(TFEstimatorBase):

            def __init__(self, input_shape, num_outputs, **kwargs):

                # Call TFEstimatorBase constructor
                super().__init__(input_shape, num_outputs, **kwargs)


            def build_model(self):

                # Model must be defined in the model attribute
                self.model = tf.keras.Sequential()
                self.model.add(layers.Dense(50, input_shape=self.input_shape, activation='relu'))
                self.model.add(layers.Dense(40, activation='relu'))
                self.model.add(layers.Dense(30, activation='relu'))
                self.model.add(layers.Dense(self.num_outputs))

                self.model.compile(
                    optimizer=tf.optimizers.Adam(),
                    loss=TFEstimatorBase.rms_error,
                    metrics=['accuracy']
                )

For a detailed example of running a synthesizer sound matching experiment using
deep learning models like this, see the :ref:`FM Sound Match Example <fm_sound_match>`.

"""

import os
from abc import abstractmethod
import datetime
import numpy as np
import tensorflow as tf

from spiegelib.estimator.estimator_base import EstimatorBase
from spiegelib.estimator.tf_epoch_logger import TFEpochLogger
from spiegelib.estimator.highway_layer import HighwayLayer

class TFEstimatorBase(EstimatorBase):
    """
    Args:
        input_shape (tuple, optional): Shape of matrix that will be passed to model input
        num_outputes (int, optional): Number of outputs the model has. If estimating
            synthesizer parameters this will typically be the number of parameters.
        weights_path (string, optional): If given, model weights will be loaded from this file
        callbacks (list, optional): A list of callbacks to be passed into model fit method

    Attributes:
        model (``tf.keras.Model``): Attribute for the model, see `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__
    """

    def __init__(self, input_shape=None, num_outputs=None,
                 weights_path = "", callbacks=[]):
        """
        Constructor
        """

        super().__init__()

        self.input_shape = input_shape
        self.num_outputs = num_outputs

        # Construct the model
        self.model = None
        self.build_model()

        # Datasets
        self.train_data = None
        self.test_data = None

        # Loggers
        if isinstance(callbacks, list):
            self.callbacks = callbacks
        else:
            raise Exception('loggers argument must be of type list, '
                            'received %s' % type(loggers))

        # Attempt to load model weights if provided
        if weights_path:
            self.load_weights(weights_path)


    @abstractmethod
    def build_model(self):
        """
        Abstract method that should contain the model definition when implemented
        """
        pass


    def add_training_data(self, input, output, batch_size=64,
                          shuffle_size=None):
        """
        Create a `tf.data.Dataset <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_
        from training data, and shuffles / batches data for training. Stores
        results in the ``train_data`` attribute.

        Args:
            input (np.ndarray): training data tensor (ex, audio features)
            output (np.ndarray): ground truth data (ex, parameter values)
            batch_size: (int, optional): If provided, will batch data into batches of
                this size. None or 0 will be no batching. Defaults to 64.
            shuffle_size (int, optional): If provided will shuffle data with a buffer
                of this size. None or 0 corresponds to no shuffling. Defaults to None.
        """

        if not input.shape[1:] == self.input_shape:
            raise Exception('Expected training data to have shape %s, got %s' \
                            % (input.shape[1:], self.input_shape))

        if not output.shape[-1] == self.num_outputs:
            raise Exception('Expected training validation data to have shape '
                            '%s, got %s' % (output.shape[-1], self.num_outputs))

        self.train_data = tf.data.Dataset.from_tensor_slices((input, output))

        if shuffle_size:
            self.train_data = self.train_data.shuffle(shuffle_size)

        if batch_size:
            self.train_data = self.train_data.batch(batch_size)


    def add_testing_data(self, input, output, batch_size=64):
        """
        Create a `tf.data.Dataset <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_
        and optionally batches for validation. Stores results in the ``test_data`` attribute.

        Args:
            input (np.ndarray): validation data (ex, audio features)
            output (np.ndarray): ground truth data (ex, parameter values)
            batch_size: (int, optional): If provided, will batch data into batches of
                this size. None or 0 will be no batching. Defaults to 64.
        """

        if not input.shape[1:] == self.input_shape:
            raise Exception('Expected training data to have shape %s, got %s' \
                            % (input.shape[1:], self.input_shape))

        if not output.shape[-1] == self.num_outputs:
            raise Exception('Expected training validation data to have shape %s, '
                             'got %s' % (output.shape[-1], self.num_outputs))

        self.test_data = tf.data.Dataset.from_tensor_slices((input, output))

        if batch_size:
            self.test_data = self.test_data.batch(batch_size)


    def fit(self, epochs=1, callbacks=[], **kwargs):
        """
        Train model on for a fixed number of epochs on training data and validation
        data if it has been added to this estimator

        Args:
            epochs (int, optional): Number of epocs to train model on.
            callbacks (list, optional): List of callback functions for training.
            kwargs: Keyword args passed to model fit method. See
                `Tensflow Docs <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.
        """

        # Check for callbacks in k
        if not isinstance(callbacks, list):
            raise Exception('Callbacks must be a list, received %s' \
                            % type(callbacks))

        # Add callbacks
        if self.callbacks:
            for callback in self.callbacks:
                callbacks.append(callback)

        # Train model
        self.model.fit(
            self.train_data,
            epochs = epochs,
            validation_data = self.test_data,
            callbacks=callbacks,
            **kwargs
        )

    def predict(self, input):
        """
        Run prediction on input

        Arg (np.ndarray): matrix of input data to run predictions on. Can a single
            instance or a batch
        """

        # If this is a single instance we need to wrap in an np.array
        is_single_input = False
        if input.shape == self.input_shape:
            is_single_input = True
            input = np.array([input])
        else:
            raise Exception('Input data has incorrect shape, expected %s, '
                             'got %s' % (self.input_shape, input.shape))

        prediction = self.model.predict(input)
        return prediction[0] if is_single_input else prediction


    def load_weights(self, filepath, **kwargs):
        """
        Load model weights from H5 or TensorFlow file

        Args:
            filepath (string): location of saved model weights
            kwargs: optional keyword arguments passed to tf load_weights
                methods, see  `TensorFlow Docs <https://www.tensorflow.org/api_docs/python/tf/keras/Model#load_weights>`__.
        """

        self.model.load_weights(filepath, **kwargs)


    def save_weights(self, filepath, **kwargs):
        """
        Save model weights to a HDF5 or TensorFlow file.

        Args:
            filepath (str): filepath to save model weights.  Using a file suffix of
                '.h5' or '.keras' will save in HDF5 format. Otherwise will save as
                TensorFlow.
            kwargs: optional keyword arguments passed to tf save_weights
                method,  see `Tensflow Docs <https://www.tensorflow.org/api_docs/python/tf/keras/Model#save_weights>`__.
        """

        path = os.path.abspath(filepath)
        dir = os.path.dirname(path)
        if not (os.path.exists(dir) and os.path.isdir(dir)):
            os.mkdir(dir)

        self.model.save_weights(path, **kwargs)


    def save_model(self, filepath, **kwargs):
        """
        Save entire model

        Args:
            filepath (str): path to SavedModel or H5 file to save the model.
            kwargs: optional keyword arguments pass to tf save method,
                see `TensorFlow Docs <https://www.tensorflow.org/api_docs/python/tf/keras/Model#save>`__.
        """

        path = os.path.abspath(filepath)
        dir = os.path.dirname(path)
        if not (os.path.exists(dir) and os.path.isdir(dir)):
            os.mkdir(dir)

        self.model.save(path, **kwargs)


    @staticmethod
    def load(filepath, **kwargs):
        """
        Load entire model and return an istantiated TFEstimatorBase class with
        the saved model loaded into it.

        Args:
            filepath (str): path to SavedModel or H5 file of saved model.
            kwargs: Keyword arguments to pass into load_model function. See
                `TensorFlow Doc <`https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model>`__.
        """

        # Add spiegelib custom objects
        custom_objects = {'rms_error': TFEstimatorBase.rms_error,
                          'HighwayLayer': HighwayLayer}

        if 'custom_objects' in kwargs:
            custom_objects.update(kwargs['custom_objects'])
            del kwargs['custom_objects']


        model = GenericTFModel()
        model.model = tf.keras.models.load_model(filepath, custom_objects=custom_objects)
        model.input_shape = model.model.get_layer(index=0).input_shape[1:]
        return model


    @staticmethod
    def rms_error(y_true, y_pred):
        """
        Static method for calculating root mean squared error between predictions
        and targets

        Args:
            y_true (Tensor): Ground truth labels
            y_pred (Tensor): Predictions
        """

        return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred))))


class GenericTFModel(TFEstimatorBase):
    """
    A generic and empty implentation TFEstimatorBase for loading models into
    """

    def build_model(self):
        """
        Empty Model
        """
        pass
