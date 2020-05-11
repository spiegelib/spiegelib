#!/usr/bin/env python
"""
Abstract Base Class for Audio Features
"""

from abc import ABC, abstractmethod
import functools
import os.path
import numpy as np
import joblib
from tqdm import trange

from spiegelib.features import StandardScaler


class FeaturesBase(ABC):
    """
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

    def __init__(self, sample_rate=44100, frame_size=2048, hop_size=512,
                 time_major=False, scale=False, scale_axis=(0,)):
        """
        Constructor
        """

        super().__init__()

        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.time_major = time_major

        self.should_scale = scale
        self.scaler = None
        self.scale_axis = scale_axis
        self.ScalerClass = StandardScaler

        self.input_modifiers = []
        self.prenorm_modifiers = []
        self.output_modifiers = []

        # Update this in inheriting classes if you need
        self.dtype = np.float32


    def __call__(self, audio, scale=None):
        """
        Calls the get_features method. Applies data modifiers prior to and after
        feature extraction

        :param audio: Audio to process features on
        :type audio: :class:`spiegelib.core.audio_buffer.AudioBuffer`
        :param normalize: If set, will override the normalize attribute set
            during construction.
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
        should_scale = scale if scale != None else self.should_scale
        if should_scale:
            assert self.has_scaler(), "Scaler must be set first."
            features = self.scale(features)

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
        elif type == 'output':
            self.output_modifiers.append(modifier)
        else:
            raise ValueError('Type must be one of ("input", "prenormalize", or '
                             '"output"), received %s' % type)


    @abstractmethod
    def get_features(self, audio):
        """
        Must be implemented. Run audio feature extraction on audio provided as parameter.
        Normalization should be applied based on the normalize parameter.

        :param audio: Audio to process features on
        :type audio: :class:`spiegelib.core.audio_buffer.AudioBuffer`
        :returns: results from audio feature extraction
        :rtype: np.array
        """
        pass


    def set_scaler(self, scaler):
        """
        Set a scaler for a dimension, this will be used to normalize that dimension

        :param scaler: A trained scaler object
        :type scaler:
        """

        self.scaler = scaler


    def fit_scaler(self, data, transform=True):
        """
        Fit scaler to dataset for future transforms.

        :param data: data to train scaler on
        :type data: np.ndarray
        :returns: np.ndarray with scaled data
        :rtype: np.ndarray
        """

        # Fit a new scaler to dataset
        self.scaler = self.ScalerClass()
        self.scaler.fit(data, self.scale_axis)

        # Scale the input dataset
        if transform:
            return self.scaler.transform(data)
        else:
            return None


    def scale(self, data):
        """
        Scale features using pre-trained scaler

        :param data: data to be normalized
        :type data: np.array
        :returns: scaled data
        :rtype: np.array
        """

        assert self.has_scaler(), "Scaler must be set first"
        return self.scaler.transform(data)


    def has_scaler(self):
        """
        :returns: a boolean indicating whether or not scaler has been set
        :rtype: boolean
        """
        return self.scaler != None


    def save_scaler(self, location):
        """
        Save the trained scaler for these features for later use

        :param location: Location to save pickled scaler
        :type location: str
        """

        joblib.dump(self.scaler, location)


    def load_scaler(self, location):
        """
        Load trained scaler from disk

        :param location: Pickled file of trained scaler
        :type location: str
        """

        self.scaler = joblib.load(location)
