#!/usr/bin/env python
"""
Abstract Base Class for Audio Features
"""

from abc import ABC, abstractmethod
import functools
import os.path
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import trange


class FeaturesBase(ABC):
    """
    :param dimensions: Number of dimensions associated with these features
    :type dimensions: int
    :param sample_rate: Audio sample rate, defaults to 44100
    :type sample_rate: int, optional
    :param frame_size: frame size in audio samples, defaults to 2048
    :type frame_size: int, optional
    :param hop_size: hop size in audio samples, defaults to 512
    :type hop_size: int, optional
    :param time_major: indicates orientation of matrix that features are returned in,
        time_major is (time_slices, features). Defaults to (features, time_slices).
    :type time_major: boolean, optional
    :param per_feature_normalize: Whether normalization should be applied to each
        dimension independently, or on entire feature space. Defaults to True, which
        applies normalization for each dimension independently.
    :type per_feature_normalize: bool, optional
    """

    def __init__(
        self,
        dimensions,
        sample_rate=44100,
        frame_size=2048,
        hop_size=512,
        time_major=False,
        per_feature_normalize=True,
    ):
        """
        Constructor
        """

        super().__init__()

        self.dimensions = dimensions
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size

        self.per_feature_normalize = per_feature_normalize
        if self.per_feature_normalize:
            self.normalizers = [None]*self.dimensions
            self.Scaler = StandardScaler
        else:
            self.normalizers = [None]
            self.Scaler = FullDataStandardScaler

        self.time_major=time_major

        self.input_modifiers = []
        self.prenorm_modifiers = []
        self.output_modifiers = []

        # Update this in inheriting classes if you need
        self.dtype = np.float32


    def __call__(self, audio, normalize=False):
        """
        Calls the get_features method. Applies data modifiers prior to and after
        feature extraction

        :param audio: Audio to process features on
        :type audio: :class:`spiegel.core.audio_buffer.AudioBuffer`
        :param normalize: Whether or not the features are normalized, defaults to False
        :type normalize: bool, optional
        :returns: results from audio feature extraction
        :rtype: np.array
        """

        # Input data modification
        for modifer in self.input_modifiers:
            audio = modifer(audio)

        # Run feature extraction
        features = self.get_features(audio)

        # Apply any prenormalization data modification
        for modifier in self.prenorm_modifiers:
            features = modifier(features)

        # Normalize features
        if normalize:
            features = self.normalize(features)

        # Apply any output data modification
        for modifier in self.output_modifiers:
            features = modifier(features)

        return features


    def add_modifier(self, modifier, type):
        """
        Add a data modifier to the feature extraction pipeline

        :param modifier: data modifier function. Should accept an np.array,
            apply some modification to that, and return a np.array
        :type modifier: lambda
        :param type: Where to add this into the pipeline. ('input', 'prenormalize', or 'output')
        :type type: str
        """

        if type == 'input':
            self.input_modifiers.append(modifier)
        elif type == 'prenormalize':
            self.prenorm_modifiers.append(modifier)
        elif type == 'outout':
            self.output_modifiers.append(modifier)
        else:
            raise ValueError(
                'Type must be one of ("input", "prenormalize", or '
                '"output"), received %s' % type
            )


    @abstractmethod
    def get_features(self, audio):
        """
        Must be implemented. Run audio feature extraction on audio provided as parameter.
        Normalization should be applied based on the normalize parameter.

        :param audio: Audio to process features on
        :type audio: :class:`spiegel.core.audio_buffer.AudioBuffer`
        :returns: results from audio feature extraction
        :rtype: np.array
        """
        pass


    def set_normalizer(self, dimension, normalizer):
        """
        Set a normalizer for a dimension, this will be used to normalize that dimension

        :param dimension: Which feature dimension to save this normalizer for
        :type dimension: int
        :param normalizer: A trained normalizer object
        :type normalizer: Sklean Scaler
        """

        self.normalizers[dimension] = normalizer


    def fit_normalizers(self, data, transform=True):
        """
        Fit normalizers to dataset for future transforms.

        :param data: data to train normalizer on
        :type data: np.ndarray
        :returns: np.ndarray with normalized data
        :rtype: np.ndarray
        """

        if not self.per_feature_normalize:
            return self.fit_normalizers_full_data(data, transform)

        elif len(data.shape) == 2:
            return self.fit_normalizers_2d(data, transform)

        elif len(data.shape) == 3:
            return self.fit_normalizers_3d(data, transform)

        else:
            raise Exception("Dimensionality of dataset not supported, only 2D or 3D matrices.")


    def fit_normalizers_full_data(self, data, transform):
        """
        Fit one normalizer on the entire dataset
        """

        scaler = self.Scaler()
        scaler.fit(data)
        scaled_data = scaler.transform(data) if transform else None
        self.set_normalizer(0, scaler)

        return scaled_data


    def fit_normalizers_2d(self, data, transform):
        """
        Fit normalizers for 2-dimensional datasets

        :param data: data to train normalizer on
        :type data: np.ndarray
        :returns: np.ndarray with normalized data
        :rtype: np.ndarray
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
        scaled_data = scaler.transform(data) if transform else None

        for i in range(self.dimensions):
            self.set_normalizer(i, scaler)

        return scaled_data


    def fit_normalizers_3d(self, data, transform):
        """
        Fit normalizers for 3-dimensional datasets

        :param data: data to train normalizer on
        :type data: np.ndarray
        :returns: np.ndarray with normalized data
        :rtype: np.ndarray
        """

        # Verify data input
        if len(data.shape) != 3:
            raise Exception("Expected 3D array for normalized data, got %s" % data.shape)

        feature_dims = 2 if self.time_major else 1
        if data.shape[feature_dims] != self.dimensions:
            raise Exception("Expected data to have %s feature dimensions, got %s" % (
                self.dimensions, data.shape[feature_dims]
            ))

        scaled_data = np.zeros_like(data) if transform else None

        # Train normalizers
        for i in trange(self.dimensions, desc="Fitting Normalizers"):
            scaler = self.Scaler()

            if self.time_major:
                scaler.fit(data[:,:,i])
                if transform:
                    scaled_data[:,:,i] = scaler.transform(data[:,:,i])
            else:
                scaler.fit(data[:,i,:])
                if transform:
                    scaled_data[:,i,:] = scaler.transform(data[:,i,:])

            self.set_normalizer(i, scaler)

        return scaled_data


    def normalize(self, data):
        """
        Normalize features using pre-trained normalizer

        :param data: data to be normalized
        :type data: np.array
        :returns: normalized data
        :rtype: np.array
        """

        normalized_data = np.zeros(data.shape, dtype=data.dtype)

        if not self.per_feature_normalize:
            normalized_data = self.normalizers[0].transform(data)

        elif len(data.shape) == 2:
            for i in range(self.dimensions):
                if not self.normalizers[i]:
                    raise NormalizerError("Normalizers not set for features. Please set normalizers first.")

                if self.time_major:
                    normalized_data[:,i] = self.normalizers[i].transform([data[:,i]])[0]
                else:
                    normalized_data[i,:] = self.normalizers[i].transform([data[i,:]])[0]

        elif len(data.shape) == 1:
            normalized_data = self.normalizers[0].transform([data])[0]

        else:
            raise ValueError('Expected 1D or 2D data, got %s' % data.shape)

        return normalized_data


    def has_normalizers(self):
        """
        :returns: a boolean indicating whether or not normalizers have been set
        :rtype: boolean
        """

        for normalizer in self.normalizers:
            if normalizer == None:
                return False

        return True

    def save_normalizers(self, location):
        """
        Save the trained normalizers for these features for later use

        :param location: Location to save pickled normalizers
        :type location: str
        """
        joblib.dump(self.normalizers, location)


    def load_normalizers(self, location):
        """
        Load trained normalizers from disk

        :param location: Pickled file of trained normalizers
        :type location: str
        """
        self.normalizers = joblib.load(location)


class FullDataStandardScaler():
    """
    Custom data scaler for working with larger dimensionality datasets that
    don't need to be scaled on a per-dimension basis such as STFT
    """

    def _reset(self):
        """
        Reset attributes
        """

        if hasattr(self, 'mean'):
            del self.mean
            del self.std


    def fit(self, data):
        """
        Compute mean and std for later scaling

        :param data: data to use to calculate mean and std on
        :type data: np.ndarray
        """

        self._reset()
        self.mean = data.mean()
        self.std = data.std()


    def transform(self, data):
        """
        Perform normalization on data

        :param data: data to normalize
        :type data: np.array
        :returns: Normalized data
        :rtype: np.ndarray
        """

        if not hasattr(self, 'mean'):
            raise Exception("You must fit this scaler first")

        return (data - self.mean) / self.std


    def fit_transform(self, data):
        """
        Compute mean and std on data and then normalize it

        :param data: data to computer mean and std on and normalize
        :type data: np.array
        :returns: Normalized data
        :rtype: np.ndarray
        """

        self.fit(data)
        return self.transform(data)


class NormalizerError(Exception):
    """
    Exception class for normalizers
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
