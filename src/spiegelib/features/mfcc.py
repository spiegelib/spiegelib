#!/usr/bin/env python
"""
Mel-Frequency Cepstral Coefficients (MFCCs)
"""

import numpy as np
import librosa
from spiegelib import AudioBuffer
from spiegelib.features.features_base import FeaturesBase

class MFCC(FeaturesBase):
    """
    Args:
        num_mfccs (int, optional): Number of MFCCs to return, defaults to 20
        frame_size (int, optional): Size of FFT to use when calculating MFCCs, defaults to 2048
        hop_size (int, optiona): hop length in samples, defaults to 512
        scale_axis (int, tuple, None): When applying scaling, determines which dimensions
            scaling be applied along. Defaults to 0, which scales each MFCC and time series
            component independently.
        kwargs: Keyword arguments, see :class:`spiegelib.features.features_base.FeaturesBase`.
    """

    def __init__(self, num_mfccs=20, frame_size=2048, hop_size=512, scale_axis=0, **kwargs):
        """
        Contructor
        """

        self.num_mfccs = num_mfccs
        self.frame_size = frame_size
        self.hop_size = hop_size
        super().__init__(scale_axis=scale_axis, **kwargs)


    def get_features(self, audio):
        """
        Run MFCC extraciton on audio buffer.

        Args:
            audio (:ref:`AudioBuffer <audio_buffer>`): input audio

        Returns:
            np.ndarray: Results of MFCC extraction. Format depends on output type set during\
                construction.
        """

        if not isinstance(audio, AudioBuffer):
            raise TypeError('audio must be AudioBuffer, recieved %s' % type(audio))

        if audio.get_sample_rate() != self.sample_rate:
            raise ValueError(
                'audio buffer samplerate does not equal feature '
                'extraction rate, %s != %s' % (audio.get_sample_rate(), self.sample_rate)
            )

        features = librosa.feature.mfcc(
            y=audio.get_audio(),
            sr=self.sample_rate,
            n_fft=self.frame_size,
            hop_length=self.hop_size,
            n_mfcc=self.num_mfccs
        )

        if self.time_major:
            features = np.transpose(features)

        return features
