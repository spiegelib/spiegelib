#!/usr/bin/env python
"""
Abstract Base Class for Audio Features
"""

from abc import ABC, abstractmethod
import os.path
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


class FeaturesBase(ABC):
    """
    :param dimensions: Number of dimensions associated with these features
    :type dimensions: int
    :param sampleRate: Audio sample rate, defaults to 44100
    :type sampleRate: int, optional
    :param frameSizeSamples: frame size in audio samples, defaults to 2048
    :type frameSizeSamples: int, optional
    :param hopSizeSamples: hop size in audio samples, defaults to 512
    :type hopSizeSamples: int, optional
    :param timeMajor: indicates orientation of matrix that features are returned in,
        timeMajor is (time_slices, features). Defaults to (features, time_slices).
    :type timeMajor: boolean, optional
    """

    def __init__(
        self,
        dimensions,
        sampleRate=44100,
        frameSize=2048,
        hopSize=512,
        timeMajor=False,
    ):
        """
        Constructor
        """

        super().__init__()

        self.dimensions = dimensions
        self.sampleRate = sampleRate
        self.frameSize = frameSize
        self.hopSize = hopSize

        self.normalizers = [None]*self.dimensions
        self.Scaler = StandardScaler

        self.timeMajor=timeMajor


    @abstractmethod
    def getFeatures(self, audio, normalize=False):
        """
        Must be implemented. Run audio feature extraction on audio provided as parameter.
        Normalization should be applied based on the normalize parameter.

        :param audio: Audio to process features on
        :type audio: np.array
        :param normalize: Whether or not the features are normalized, defaults to False
        :type normalize: bool, optional
        :returns: results from audio feature extraction
        :rtype: np.array
        """
        pass


    def setNormalizer(self, dimension, normalizer):
        """
        Set a normalizer for a dimension, this will be used to normalize that dimension

        :param dimension: Which feature dimension to save this normalizer for
        :type dimension: int
        :param normalizer: A trained normalizer object
        :type normalizer: Sklean Scaler
        """

        self.normalizers[dimension] = normalizer


    def fitNormalizers(self, data, transform=False):
        """
        Fit normalizers to dataset for future transforms. Can also transform
        the data and return a normalized version of that data.

        :param data: data to train normalizer on
        :type data: np.array
        :param transform: should the incoming data also be normalized? Defaults to False
        :type transform: bool, optional
        :returns: None if no transform applied, np.array with normalized data if transform applied
        :rtype: None or np.array
        """

        if len(data.shape) == 2:
            self.fitNormalizers2D(data, transform)

        elif len(data.shape) == 3:
            self.fitNormalizers3D(data, transform)

        else:
            raise Exception("Dimensionality of dataset not supported, only 2D or 3D matrices.")


    def fitNormalizers2D(self, data, transform=False):
        """
        Fit normalizers for 2-dimensional datasets

        :param data: data to train normalizer on
        :type data: np.array
        :param transform: should the incoming data also be normalized? Defaults to False
        :type transform: bool, optional
        :returns: None if no transform applied, np.array with normalized data if transform applied
        :rtype: None or np.array
        """

        # Verify data input
        if len(data.shape) != 2:
            raise Exception("Expected 2D array for normalized data, got %s" % data.shape)

        if data.shape[1] != self.dimensions:
            raise Exception("Expected data to have %s feature dimensions, got %s" % (
                self.dimensions, data.shape[1]
            ))

        scaler = self.Scaler()
        scaler.fit(data)
        if transform:
            scaledData = scaler.transform(data)

        for i in range(self.dimensions):
            self.setNormalizer(i, scaler)

        return scaledData if transform else None


    def fitNormalizers3D(self, data, transform=False):
        """
        Fit normalizers for 3-dimensional datasets

        :param data: data to train normalizer on
        :type data: np.array
        :param transform: should the incoming data also be normalized? Defaults to False
        :type transform: bool, optional
        :returns: None if no transform applied, np.array with normalized data if transform applied
        :rtype: None or np.array
        """

        # Verify data input
        if len(data.shape) != 3:
            raise Exception("Expected 3D array for normalized data, got %s" % data.shape)

        featureDimension = 2 if self.timeMajor else 1
        if data.shape[featureDimension] != self.dimensions:
            raise Exception("Expected data to have %s feature dimensions, got %s" % (
                self.dimensions, data.shape[featureDimension]
            ))

        if transform:
            scaledData = np.zeros_like(data)
        else:
            scaledData = None

        # Train normalizers
        for i in range(self.dimensions):
            scaler = self.Scaler()

            if self.timeMajor:
                scaler.fit(data[:,:,i])
            else:
                scaler.fit(data[:,i,:])

            if transform:
                if self.timeMajor:
                    scaledData[:,:,i] = scaler.transform(data[:,:,i])
                else:
                    scaledData[:,i,:] = scaler.transform(data[:,i,:])

            self.setNormalizer(i, scaler)

        return scaledData


    def normalize(self, data):
        """
        Normalize features using pre-trained normalizer

        :param data: data to be normalized
        :type data: np.array
        :returns: normalized data
        :rtype: np.array
        """

        normalizedData = np.zeros(data.shape, dtype=np.float32)

        if len(data.shape) == 2:
            for i in range(self.dimensions):
                if not self.normalizers[i]:
                    raise NormalizerError("Normalizers not set for features. Please set normalizers first.")
                normalizedData[i,:] = self.normalizers[i].transform([data[i,:]])[0]

        elif len(data.shape) == 1:
            normalizedData = self.normalizers[0].transform([data])[0]

        else:
            raise ValueError('Expected 1D or 2D data, got %s' % data.shape)

        return normalizedData


    def hasNormalizers(self):
        """
        :returns: a boolean indicating whether or not normalizers have been set
        :rtype: boolean
        """

        for normalizer in self.normalizers:
            if normalizer == None:
                return False

        return True



    def saveNormalizers(self, location):
        """
        Save the trained normalizers for these features for later use

        :param location: Location to save pickled normalizers
        :type location: str
        """
        joblib.dump(self.normalizers, location)


    def loadNormalizers(self, location):
        """
        Load trained normalizers from disk

        :param location: Pickled file of trained normalizers
        :type location: str
        """
        self.normalizers = joblib.load(location)



class NormalizerError(Exception):
    """
    Exception class for normalizers
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
