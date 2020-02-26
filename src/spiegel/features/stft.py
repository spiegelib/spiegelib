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
    :param fftSize: number of FFT bins, defaults to 1024 (overrides frameSize)
    :type fftSize: int, optional
    :param kwargs: keyword arguments for base class, see :class:`spiegel.features.features_base.FeaturesBase`.
    """

    def __init__(self, fftSize=1024, **kwargs):
        """
        Contructor
        """

        dims = int(1 + (fftSize / 2))
        super().__init__(dims, perFeatureNormalize=False, **kwargs)
        self.frameSize = fftSize
        self.dtype = np.complex64


    def getFeatures(self, audio, normalize=False):
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
            y=audio.getAudio(),
            n_fft=self.frameSize,
            hop_length=self.hopSize,
        )

        if self.timeMajor:
            features = np.transpose(features)

        return features
