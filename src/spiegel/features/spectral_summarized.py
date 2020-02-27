#!/usr/bin/env python
"""
Spectral features summarized over time using mean and variance
"""

import numpy as np
import librosa
from spiegel import AudioBuffer
from spiegel.features.features_base import FeaturesBase

class SpectralSummarized(FeaturesBase):
    """
    :param kwargs: See :class:`spiegel.features.features_base.FeaturesBase`
    """

    def __init__(self, **kwargs):
        """
        Constructor
        """
        dimensions = 10
        super().__init__(dimensions, **kwargs)


    def getFeatures(self, audio, normalize=False):
        """
        Run audio feature extraction on audio provided as parameter.
        Normalization should be applied based on the normalize parameter.

        :param audio: Audio to process features on
        :type audio: np.array
        :param normalize: Whether or not the features are normalized, defaults to False
        :type normalize: bool, optional
        :returns: results from audio feature extraction
        :rtype: np.array
        """

        if not isinstance(audio, AudioBuffer):
            raise TypeError('audio must be AudioBuffer, recieved %s' % type(audio))

        if audio.get_sample_rate() != self.sampleRate:
            raise ValueError(
                'audio buffer samplerate does not equal feature '
                'extraction rate, %s != %s' % (audio.get_sample_rate(), self.sampleRate)
            )

        spectralCentroid = librosa.feature.spectral_centroid(
            y=audio.get_audio(),
            sr=self.sampleRate,
            n_fft=self.frameSize,
            hop_length=self.hopSize,
        )

        spectralBandwidth = librosa.feature.spectral_bandwidth(
            y=audio.get_audio(),
            sr=self.sampleRate,
            n_fft=self.frameSize,
            hop_length=self.hopSize,
        )

        spectralContrast = librosa.feature.spectral_contrast(
            y=audio.get_audio(),
            sr=self.sampleRate,
            n_fft=self.frameSize,
            hop_length=self.hopSize,
        )

        spectralFlatness = librosa.feature.spectral_flatness(
            y=audio.get_audio(),
            n_fft=self.frameSize,
            hop_length=self.hopSize,
        )

        spectralRolloff = librosa.feature.spectral_rolloff(
            y=audio.get_audio(),
            sr=self.sampleRate,
            n_fft=self.frameSize,
            hop_length=self.hopSize,
        )

        features = np.array([
            spectralCentroid.mean(),
            spectralCentroid.var(),
            spectralBandwidth.mean(),
            spectralBandwidth.var(),
            spectralContrast.mean(),
            spectralContrast.var(),
            spectralFlatness.mean(),
            spectralFlatness.var(),
            spectralRolloff.mean(),
            spectralRolloff.var()
        ])

        return features
