#!/usr/bin/env python
"""
Abstract Base Class for Estimating Synthesizer Parameters using TensorFlow
"""

import os
from abc import abstractmethod
from spiegel.estimator.estimator_base import EstimatorBase
import numpy as np
import tensorflow as tf

class TFEstimatorBase(EstimatorBase):
    """
    :param inputShape: Shape of matrix that will be passed to model input
    :type inputShape: tuple
    :param numOutputs: Number of outputs the model has
    :type numOuputs: int
    :param checkpointPath: If given, checkpoints will be saved to this location
        during training, defaults to ""
    :type checkpointPath: string, optional
    :param weightsPath: If given, model weights will be loaded from this file,
        defaults to ""
    :type weightsPath: string, optional
    """

    def __init__(self, inputShape, numOutputs, checkpointPath = "", weightsPath = ""):
        """
        Constructor
        """

        super().__init__()

        self.inputShape = inputShape
        self.numOutputs = numOutputs

        # Construct the model
        self.model = None
        self.buildModel()

        # Datasets
        self.trainData = None
        self.testData = None

        # Checkpoints
        self.checkpointPath = None
        self.checkpointDir = None
        if checkpointPath:
            self.checkpointPath = os.path.abspath(checkpointPath)
            self.checkpointDir = os.path.dirname(self.checkpointPath)
            if os.path.exists(self.checkpointDir):
                self.loadModelFromCheckpoint()

        # Attempt to load model weights if provided
        if weightsPath:
            self.loadWeights(weightsPath)


    def addTrainingData(self, input, output, batchSize=64, shuffleSize=None):
        """
        Create a tf Dataset from input and output, and shuffles / batches data for training

        :param input: matrix of training data
        :type input: np.array
        :param output: matrix of training data ground truth
        :type output: np.array
        :param batchSize: If provided, will batch data into batches of this size,
            set to None or 0 to prevent batching. defaults to 64
        :type batchSize: int, optional
        :param shuffleSize: If provided, will shuffle data with a buffer size of
            shuffleSize, defaults to None, so shuffling does not occur
        :type batchSize: int, optional
        """

        if not input.shape[1:] == self.inputShape:
            raise Exception('Expected training data to have shape %s, got %s' % (input.shape[1:], self.inputShape))

        if not output.shape[-1] == self.numOutputs:
            raise Exception('Expected training validation data to have shape %s, got %s' % (output.shape[-1], self.numOutputs))

        self.trainData = tf.data.Dataset.from_tensor_slices((input, output))

        if shuffleSize:
            self.trainData = self.trainData.shuffle(shuffleSize)

        if batchSize:
            self.trainData = self.trainData.batch(batchSize)


    def addTestingData(self, input, output, batchSize=64):
        """
        Create a tf Dataset from input and output for model testing, batches data if desired

        :param input: matrix of data to use as testing data
        :type input: np.array
        :param output: matrix of data to use as ground truth for testing data
        :type output: np.array
        :param batchSize: If provided, will batch data into batches of this size,
            set to None or 0 to prevent batching. defaults to 64
        :type batchSize: int, optional
        """

        if not input.shape[1:] == self.inputShape:
            raise Exception('Expected training data to have shape %s, got %s' % (input.shape[1:], self.inputShape))

        if not output.shape[-1] == self.numOutputs:
            raise Exception('Expected training validation data to have shape %s, got %s' % (output.shape[-1], self.numOutputs))

        self.testData = tf.data.Dataset.from_tensor_slices((input, output))

        if batchSize:
            self.testData = self.testData.batch(batchSize)


    @abstractmethod
    def buildModel(self):
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
        if input.shape == self.inputShape:
            input = np.array([input])
        else:
            raise Exception('Input data has incorrect shape, expected %s, got %s' % (self.inputShape, input.shape))

        return self.model.predict(input)


    def fit(self, epochs=1, **kwargs):
        """
        Train model on for a fixed number of epochs on training data and validation
        data if it has been added to this estimator

        :param epochs: Number of epochs to train model on, defaults to 1
        :type epochs: int, optional
        :param kwargs: Keyword args passed to model fit method. See `Tensflow Docs <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.
        """

        # Add checkpoint callback to save weights if checkpoint path has been set
        callbacks = []
        if self.checkpointPath:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.checkpointPath,
                    save_weights_only=True,
                    verbose=1
                )
            )

        # Train model
        self.model.fit(
            self.trainData,
            epochs = epochs,
            validation_data = self.testData,
            callbacks=callbacks,
            **kwargs
        )


    def loadModelFromCheckpoint(self):
        """
        Load model weights from checkpoint
        """

        latest = tf.train.latest_checkpoint(self.checkpointDir)
        self.model.load_weights(latest)


    def loadWeights(self, filepath, **kwargs):
        """
        Load model weights from H5 or TensorFlow file

        :param filepath: filepath to saved model weights
        :type filepath: string
        :param kwargs: optional keyword arguments passed to tf load_weights methods, see  `TensorFlow Docs <https://www.tensorflow.org/api_docs/python/tf/keras/Model#load_weights>`__.
        """

        self.model.load_weights(filepath, **kwargs)


    def saveWeights(self, filepath, **kwargs):
        """
        Save model weights to a HDF5 or TensorFlow file.

        :param filepath: filepath to save model weights.  Using a file suffix of
            '.h5' or '.keras' will save in HDF5 format. Otherwise will save as TensorFlow.
        :type filepath: string
        :param kwargs: optional keyword arguments passed to tf save_weights method,  see `Tensflow Docs <https://www.tensorflow.org/api_docs/python/tf/keras/Model#save_weights>`__.
        """

        path = os.path.abspath(filepath)
        dir = os.path.dirname(path)
        if not (os.path.exists(dir) and os.path.isdir(dir)):
            os.mkdir(dir)

        self.model.save_weights(path, **kwargs)


    @staticmethod
    def rootMeanSquaredError(labels, prediction):
        """
        Static method for calculating root mean squared error between predictions and targets

        :param labels: Matrix of ground truth labels
        :type labels: Tensor
        :param prediction: Matrix of predictions
        :type prediction: Tensor
        """

        return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(labels, prediction))))
