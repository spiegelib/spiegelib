#!/usr/bin/env python
"""
STFT Audio Feature Extractor
"""

import numpy as np
import librosa
from spiegelib import AudioBuffer
from spiegelib.features.features_base import FeaturesBase
import spiegelib.core.utils as utils

class STFT(FeaturesBase):
    """
    :param fft_size: number of FFT bins, defaults to 1024 (overrides frame_size)
    :type fft_size: int, optional
    :param kwargs: keyword arguments for base class, see :class:`spiegelib.features.features_base.FeaturesBase`.
    """

    def __init__(self, fft_size=1024, output='complex', scale_axis=None, **kwargs):
        """
        Contructor
        """

        super().__init__(scale_axis=scale_axis, **kwargs)

        if output not in utils.spectrum_types:
            raise TypeError('output must be one of %s' % utils.spectrum_types)

        self.output = output
        self.frame_size = fft_size
        self.dtype = np.float32
        self.complex_dtype = np.complex64


    def get_features(self, audio):
        """
        Run audio feature extraction on audio provided as parameter.
        Normalization should be applied based on the normalize parameter.

        :param audio: Audio to process features on
        :type audio: :class:`spiegelib.core.audio_buffer.AudioBuffer`
        :param normalize: Whether or not the features are normalized, defaults to False
        :type normalize: bool, optional
        :returns: results from audio feature extraction
        :rtype: np.array
        """

        if not isinstance(audio, AudioBuffer):
            raise TypeError('audio must be AudioBuffer, recieved %s' % type(audio))

        features = librosa.stft(
            y=audio.get_audio(),
            n_fft=self.frame_size,
            hop_length=self.hop_size,
        )

        features = utils.convert_spectrum(features, self.output, dtype=self.dtype,
                                          complex_dtype=self.complex_dtype)

        if self.time_major:
            features = np.swapaxes(features, 0, 1)

        return features
