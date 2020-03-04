#!/usr/bin/env python
"""
Abstract Base Class for Estimating Synthesizer Parameters using TensorFlow
"""

import os
from abc import abstractmethod
import datetime
from spiegel.estimator.estimator_base import EstimatorBase
from spiegel.estimator.tf_epoch_logger import TFEpochLogger
from spiegel.estimator.highway_layer import HighwayLayer
import numpy as np
import tensorflow as tf

class TFEstimatorBase(EstimatorBase):
    """
    :param input_shape: Shape of matrix that will be passed to model input
    :type input_shape: tuple
    :param num_outputs: Number of outputs the model has
    :type num_outputs: int
    :param checkpoint_path: If given, checkpoints will be saved to this location
        during training, defaults to ""
    :type checkpoint_path: string, optional
    :param weights_path: If given, model weights will be loaded from this file,
        defaults to ""
    :type weights_path: string, optional
    :param callbacks: A list of callbacks to be passed into model fit method,
        defaults to []
    :type callbacks: list
    """

    def __init__(self, input_shape=None, num_outputs=None, checkpoint_path = "",
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

        # Checkpoints
        self.checkpoint_path = None
        self.checkpoint_dir = None
        if checkpoint_path:
            self.checkpoint_path = os.path.abspath(checkpoint_path)
            self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
            if os.path.exists(self.checkpoint_dir):
                self.load_model_from_checkpoint()

        # Attempt to load model weights if provided
        if weights_path:
            self.load_weights(weights_path)


    def add_training_data(self, input, output, batch_size=64,
                          shuffle_size=None):
        """
        Create a tf Dataset from input and output, and shuffles / batches
        data for training

        :param input: matrix of training data
        :type input: np.ndarray
        :param output: matrix of training data ground truth
        :type output: np.ndarray
        :param batch_size: If provided, will batch data into batches of this size,
            set to None or 0 to prevent batching. defaults to 64
        :type batch_size: int, optional
        :param shuffle_size: If provided, will shuffle data with a buffer size of
            shuffle_size, defaults to None, so shuffling does not occur
        :type batch_size: int, optional
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
        Create a tf Dataset from input and output for model testing, batches
        data if desired

        :param input: matrix of data to use as testing data
        :type input: np.array
        :param output: matrix of data to use as ground truth for testing data
        :type output: np.array
        :param batch_size: If provided, will batch data into batches of this size,
            set to None or 0 to prevent batching. defaults to 64
        :type batch_size: int, optional
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


    @abstractmethod
    def build_model(self):
        """
        Abstract method that should contain the model definition when implemented
        """
        pass


    def predict(self, input):
        """
        Run prediction on input

        :param input: matrix of input data to run predictions on. Can be a single
            instance of data or a batch.
        :type input: np.array
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


    def fit(self, epochs=1, callbacks=[], **kwargs):
        """
        Train model on for a fixed number of epochs on training data and validation
        data if it has been added to this estimator

        :param epochs: Number of epochs to train model on, defaults to 1
        :type epochs: int, optional
        :param callbacks: List of callback functions for training, defaults to []
        :type callbacks: list, optional
        :param kwargs: Keyword args passed to model fit method. See
            `Tensflow Docs <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.
        """

        # Check for callbacks in k
        if not isinstance(callbacks, list):
            raise Exception('Callbacks must be a list, received %s' \
                            % type(callbacks))

        # Add a checkpoint callback if the checkpoint path has been set
        if self.checkpoint_path:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.checkpoint_path,
                    save_weights_only=True,
                    verbose=1
                )
            )

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

    def load_model_from_checkpoint(self):
        """
        Load model weights from checkpoint
        """

        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        self.model.load_weights(latest)


    def load_weights(self, filepath, **kwargs):
        """
        Load model weights from H5 or TensorFlow file

        :param filepath: filepath to saved model weights
        :type filepath: string
        :param kwargs: optional keyword arguments passed to tf load_weights
            methods, see  `TensorFlow Docs <https://www.tensorflow.org/api_docs/python/tf/keras/Model#load_weights>`__.
        """

        self.model.load_weights(filepath, **kwargs)


    def save_weights(self, filepath, **kwargs):
        """
        Save model weights to a HDF5 or TensorFlow file.

        :param filepath: filepath to save model weights.  Using a file suffix of
            '.h5' or '.keras' will save in HDF5 format. Otherwise will save as
            TensorFlow.
        :type filepath: string
        :param kwargs: optional keyword arguments passed to tf save_weights
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

        :param filepath: path to SavedModel or H5 file to save the model.
        :type filepath: str
        :param kwargs: optional keyword arguments pass to tf save method,
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

        :param filepath: path to SavedModel or H5 file of saved model.
        :type filepath: str
        :param kwargs: Keyword arguments to pass into load_model function. See
            `TensorFlow Doc <`https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model>`__.
        """

        # Add spiegel custom objects
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

        :param labels: Matrix of ground truth labels
        :type labels: Tensor
        :param prediction: Matrix of predictions
        :type prediction: Tensor
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
