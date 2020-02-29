#!/usr/bin/env python
"""
STFT Audio Feature Extractor
"""

import numpy as np
import librosa
from spiegel import AudioBuffer
from spiegel.features.features_base import FeaturesBase

class STFT(FeaturesBase):
    """
    :param fft_size: number of FFT bins, defaults to 1024 (overrides frame_size)
    :type fft_size: int, optional
    :param kwargs: keyword arguments for base class, see :class:`spiegel.features.features_base.FeaturesBase`.
    """

    def __init__(self, fft_size=1024, **kwargs):
        """
        Contructor
        """

        dims = int(1 + (fft_size / 2))
        super().__init__(dims, per_feature_normalize=False, **kwargs)
        self.frame_size = fft_size
        self.dtype = np.complex64


    def get_features(self, audio):
        """
        Run audio feature extraction on audio provided as parameter.
        Normalization should be applied based on the normalize parameter.

        :param audio: Audio to process features on
        :type audio: :class:`spiegel.core.audio_buffer.AudioBuffer`
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

        if self.time_major:
            features = np.transpose(features)

        return features
