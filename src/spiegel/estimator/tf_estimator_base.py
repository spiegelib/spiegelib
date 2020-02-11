#!/usr/bin/env python
"""
Abstract Base Class for Estimating Synthesizer Parameters using TensorFlow
"""

import os
from abc import abstractmethod
from spiegel.estimator.estimator_base import EstimatorBase
import tensorflow as tf

class TFEstimatorBase(EstimatorBase):

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


    def addTrainingData(self, input, output, shuffleSize=100, batchSize=64):
        """
        Create a tf Dataset from input and output, and shuffles / batches data for training
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
        """

        return self.model.predict(input)


    def fit(self, epochs=1, **kwargs):
        """
        Train model on for a fixed number of epochs on training data and validation
        data if it has been added to this estimator
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
        latest = tf.train.latest_checkpoint(self.checkpointDir)
        self.model.load_weights(latest)


    def loadWeights(self, filepath, **kwargs):
        self.model.load_weights(filepath, **kwargs)


    def saveWeights(self, filepath, **kwargs):
        path = os.path.abspath(filepath)
        dir = os.path.dirname(path)
        if not (os.path.exists(dir) and os.path.isdir(dir)):
            os.mkdir(dir)

        self.model.save_weights(path, **kwargs)



    @staticmethod
    def rootMeanSquaredError(labels, prediction):
        """
        Static method for calculating root mean squared error between predictions and targets
        """
        return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(labels, prediction))))
