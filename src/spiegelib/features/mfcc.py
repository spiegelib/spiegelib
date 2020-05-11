#!/usr/bin/env python
"""
MFCC Audio Feature Extractor
"""

import numpy as np
import librosa
from spiegelib import AudioBuffer
from spiegelib.features.features_base import FeaturesBase

class MFCC(FeaturesBase):
    """
    :param num_mfccs: number of mffcs to return per frame, defaults to 20
    :type num_mfccs: int, optional
    :param kwargs: keyword arguments for base class, see :class:`spiegelib.features.features_base.FeaturesBase`.
    """

    def __init__(self, num_mfccs=20, scale_axis=0, **kwargs):
        """
        Contructor
        """

        self.num_mfccs = num_mfccs
        super().__init__(scale_axis=scale_axis, **kwargs)


    def get_features(self, audio, normalize=False):
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
