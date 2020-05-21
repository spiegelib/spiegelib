#!/usr/bin/env python
"""
Abstract Base Class for audio feature extraction. Defines an interface for
audio feature extraction algorithms.

Inheriting classes must override :py:meth:`~spiegelib.features.FeaturesBase.get_features` and implement feature
extraction in that function, which accepts an :ref:`AudioBuffer <audio_buffer>` and
must return a `np.ndarray` containing results.

Feature extraction is run in inheriting classes through the __call__ member function.

Example of FFT, which inherits from FeaturesBase::

    audio = AudioBuffer('./some_audio.wav')
    fft = FFT()

    # Now feature extraction is run by treating
    # the FFT instance, fft, like a function
    spectrum = fft(audio)
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
    Args:
        sample_rate (int, optional): audio sample rate, defaults to 44100
        time_major (bool, optional): for audio feature extraction that uses time slices,
            used to indicate the orientation of features and time slices in the resulting
            matrix. True indicates that results should be returned with an orientation
            like (time_slices, features). False (default) refers to an orientation
            of (features, time_slices).
        scale (bool, optional): whether to scale the results of feature extraction. The scaler
            must be set before scaling called be applied. See :py:meth:`~spiegelib.features.FeaturesBase.fit_scaler` or
            :py:meth:`~spiegelib.features.FeaturesBase.set_scaler`.
        scale_axis (int, tuple, None, optional): indicates the axis to use for fitting scalers and
            applying data scaling. A value of None flattens the dataset and calculates
            scaling parameters on that. A value of an int or a tuple indicates the axis
            or axes to use. Defaults to (0,) which causes scaling variables to be calculated
            on each feature independently.
    """

    def __init__(self, sample_rate=44100, time_major=False, scale=False, scale_axis=(0,)):
        """
        Constructor
        """

        super().__init__()

        self.sample_rate = sample_rate
        self.time_major = time_major

        self.should_scale = scale
        self.scaler = None
        self.scale_axis = scale_axis
        self.ScalerClass = StandardScaler

        self.input_modifiers = []
        self.prescale_modifiers = []
        self.output_modifiers = []

        # Update this in inheriting classes if you need
        self.dtype = np.float32


    def __call__(self, audio, scale=None):
        """
        Run this feature extraction pipeline.

        Applies functions in this order: input modifiers > feature extraction >
        prescale modifiers > data scaling > output modifiers

        Args:
            audio (:ref:`AudioBuffer <audio_buffer>`): Audio to extract features from
            scale (bool, optional): If set, will override scale attribute set
                during construction.

        Returns:
            np.ndarray: results from audio feature extraction with modifiers and scaling.
        """

        # Input data modification
        for modifer in self.input_modifiers:
            audio = modifer(audio)

        # Run feature extraction
        features = self.get_features(audio)

        # Apply any prescaling data modification
        for modifier in self.prescale_modifiers:
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


    @abstractmethod
    def get_features(self, audio):
        """
        Must be implemented. Run audio feature extraction on audio provided as parameter.
        Must check the `time_major` attribute and return data in the correct
        orientation.

        Args:
            audio (:ref:`AudioBuffer <audio_buffer>`): Audio to extract features from

        Returns:
            np.ndarray: Results of audio feature extraction
        """
        pass


    def add_modifier(self, modifier, type):
        """
        Add a data modifier to the feature extraction pipeline.

        Input modifiers are applied to raw `AudioBuffers <audio_buffer>` prior
        to feature extraction. Prescale modifiers are applied to results of audio
        feature extraction and before data scaling (if applicable). Output modifiers
        are applied after data scaling.

        Args:
            modifier (lambda): data modifier function. Should accept an np.array,
                apply some modification to that, and return a np.array
            type (str): Where to add function into pipeline. Must be one of
                ('input', 'prescale', or 'output')
        """

        if type == 'input':
            self.input_modifiers.append(modifier)
        elif type == 'prescale':
            self.prescale_modifiers.append(modifier)
        elif type == 'output':
            self.output_modifiers.append(modifier)
        else:
            raise ValueError('type must be one of ("input", "prescale", or '
                             '"output"), received: %s' % type)


    def fit_scaler(self, data, transform=True):
        """
        Fit scaler to dataset for future transforms.

        Args:
            data (np.ndarray): data to train (fit) scaler on
            transform (bool, optional): if True will also apply scaling
                to the data

        Returns:
            np.ndarray, None: Scaled data if transform parameter is True, otherwise\
                returns None.
        """

        # Fit a new scaler to dataset
        self.scaler = self.ScalerClass()
        self.scaler.fit(data, self.scale_axis)

        # Scale the input dataset
        if transform:
            return self.scaler.transform(data)
        else:
            return None


    def has_scaler(self):
        """
        Returns:
            bool: whether or not the scaler has been set
        """
        return self.scaler != None


    def set_scaler(self, scaler):
        """
        Set a scaler for a dimension, this will be used to normalize that dimension

        Args:
            scaler (object): a Scaler object to use to scale data. Must be a
                :ref:`DataScalerBase <data_scaler_base>` type to use the :py:meth:`~spiegelib.features.FeaturesBase.fit_scaler`
                method. Otherwise, other scalers like sklearn scalers could potentially
                be fit and then passed in here -- just needs to have a transform method
                to apply scaling. (note, using other objects outside of :ref:`DataScalerBase <data_scaler_base>` type
                has not been tested!)
        """

        self.scaler = scaler


    def scale(self, data):
        """
        Scale features using pre-trained scaler

        Args:
            data (np.ndarray): data to be scaled

        Returns:
            np.ndarray: scaled data
        """

        assert self.has_scaler(), "Scaler must be set first"
        return self.scaler.transform(data)


    def load_scaler(self, location):
        """
        Load trained scaler from a pickled file.

        Args:
            location (str): Location of pickled scaler object
        """

        self.scaler = joblib.load(location)


    def save_scaler(self, location):
        """
        Save the trained scaler for these features as a pickle for later use

        Args:
            location (str): Location to save pickled scaler object
        """

        joblib.dump(self.scaler, location)
