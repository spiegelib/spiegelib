#!/usr/bin/env python
"""
Spectral features summarized over time using mean and variance. Returns a 22-dimension
feature vector for each audio sample.

Features:
    - Spectral Centroid
    - Spectral Bandwidth
    - Spectral Contrast (7 frequency bands)
    - Spectral Flatness
    - Spectral Rolloff
"""

import numpy as np
import librosa
from spiegelib import AudioBuffer
from spiegelib.features.features_base import FeaturesBase

class SpectralSummarized(FeaturesBase):
    """
    Args:
        frame_size (int, optional): size of FFT, defaults to 2048
        hop_size (int, optional): size of hop shift in samples, defuault to 512
        scale_axis (int, tuple, None): When applying scaling, determines which dimensions
            scaling be applied along. Defaults to 0, which scales each feature independently.
        kwargs: Keyword arguments, see :class:`spiegelib.features.features_base.FeaturesBase`.
    """

    def __init__(self, frame_size=2048, hop_size=512, scale_axis=0, **kwargs):
        """
        Constructor
        """

        self.frame_size = frame_size
        self.hop_size = hop_size
        super().__init__(scale_axis=scale_axis, **kwargs)


    def get_features(self, audio):
        """
        Extract spectral features and return results.

        Args:
            audio (:ref:`AudioBuffer <audio_buffer>`): input audio

        Returns:
            np.ndarray: Results of spectral features extraction. Format depends on\
            output type set during construction.
        """

        if not isinstance(audio, AudioBuffer):
            raise TypeError('audio must be AudioBuffer, recieved %s' % type(audio))

        if audio.get_sample_rate() != self.sample_rate:
            raise ValueError(
                'audio buffer samplerate does not equal feature '
                'extraction rate, %s != %s' % (audio.get_sample_rate(), self.sample_rate)
            )

        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio.get_audio(),
            sr=self.sample_rate,
            n_fft=self.frame_size,
            hop_length=self.hop_size,
        )

        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio.get_audio(),
            sr=self.sample_rate,
            n_fft=self.frame_size,
            hop_length=self.hop_size,
        )

        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio.get_audio(),
            sr=self.sample_rate,
            n_fft=self.frame_size,
            hop_length=self.hop_size,
        )

        spectral_flatness = librosa.feature.spectral_flatness(
            y=audio.get_audio(),
            n_fft=self.frame_size,
            hop_length=self.hop_size,
        )

        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio.get_audio(),
            sr=self.sample_rate,
            n_fft=self.frame_size,
            hop_length=self.hop_size,
        )

        features = np.array([
            spectral_centroid.mean(),
            spectral_centroid.var(),
            spectral_bandwidth.mean(),
            spectral_bandwidth.var(),
            spectral_flatness.mean(),
            spectral_flatness.var(),
            spectral_rolloff.mean(),
            spectral_rolloff.var(),
            spectral_contrast[0].mean(),
            spectral_contrast[0].var(),
            spectral_contrast[1].mean(),
            spectral_contrast[1].var(),
            spectral_contrast[2].mean(),
            spectral_contrast[2].var(),
            spectral_contrast[3].mean(),
            spectral_contrast[3].var(),
            spectral_contrast[4].mean(),
            spectral_contrast[4].var(),
            spectral_contrast[5].mean(),
            spectral_contrast[5].var(),
            spectral_contrast[6].mean(),
            spectral_contrast[6].var(),
        ])

        return features
