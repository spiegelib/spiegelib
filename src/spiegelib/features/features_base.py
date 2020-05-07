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

    def __init__(
        self,
        sample_rate=44100,
        frame_size=2048,
        hop_size=512,
        normalize=False,
        time_major=False,
        per_feature_normalize=True,
        uses_time_slices=None,
        scale_axis=None,
        scale_axis_time_major=None
    ):
        """
        Constructor
        """

        super().__init__()

        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.should_normalize = normalize

        self.per_feature_normalize = per_feature_normalize
        self.normalizer = None
        self.Scaler = StandardScaler


        if uses_time_slices == None:
            raise NotImplementedError(("Inheriting classes must specify whether "
                                       "results are returned with time slices or not"))
        self.uses_time_slices = uses_time_slices

        if scale_axis == None:
            raise NotImplementedError(("Inheriting classes must specify a scale axis "
                                       "to scale results along"))
        self.scale_axis = scale_axis

        if uses_time_slices and scale_axis_time_major == None:
            raise NotImplementedError(("Inheriting classes must spcify a scale axis "
                                       "to scale results along for time major results"))
        self.scale_axis_time_major = scale_axis_time_major


        self.time_major = time_major

        self.input_modifiers = []
        self.prenorm_modifiers = []
        self.output_modifiers = []

        # Update this in inheriting classes if you need
        self.dtype = np.float32


    def __call__(self, audio, normalize=None):
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
        shouldNormalize = normalize if normalize != None else self.should_normalize
        if shouldNormalize:
            assert self.has_normalizers(), "Normalizers must be set first."
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
        elif type == 'output':
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
        :type audio: :class:`spiegelib.core.audio_buffer.AudioBuffer`
        :returns: results from audio feature extraction
        :rtype: np.array
        """
        pass


    def set_normalizer(self, normalizer):
        """
        Set a normalizer for a dimension, this will be used to normalize that dimension

        :param normalizer: A trained normalizer object
        :type normalizer:
        """

        self.normalizer = normalizer


    def fit_normalizers(self, data, transform=True):
        """
        Fit normalizers to dataset for future transforms.

        :param data: data to train normalizer on
        :type data: np.ndarray
        :returns: np.ndarray with normalized data
        :rtype: np.ndarray
        """

        # Fit a new normalizer to dataset
        self.normalizer = self.Scaler()
        if not self.per_feature_normalize:
            self.normalizer.fit(data)

        elif self.uses_time_slices and self.time_major:
            self.normalizer.fit(data, self.scale_axis_time_major)

        else:
            self.normalizer.fit(data, self.scale_axis)

        # Scale the input dataset
        if transform:
            return self.normalizer.transform(data)
        else:
            return None


    def normalize(self, data):
        """
        Normalize features using pre-trained normalizer

        :param data: data to be normalized
        :type data: np.array
        :returns: normalized data
        :rtype: np.array
        """

        if self.normalizer == None:
            raise Exception("Normalizer must be fit first")

        return self.normalizer.transform(data)


    def has_normalizers(self):
        """
        :returns: a boolean indicating whether or not normalizers have been set
        :rtype: boolean
        """
        return self.normalizer != None

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
