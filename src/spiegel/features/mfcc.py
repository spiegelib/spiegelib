#!/usr/bin/env python
"""
MFCC Audio Feature Extractor
"""

import numpy as np
import librosa
from spiegel import AudioBuffer
from spiegel.features.features_base import FeaturesBase

class MFCC(FeaturesBase):
    """
    :param numMFCCs: number of mffcs to return per frame, defaults to 20
    :type numMFCCs: int, optional
    :param sampleRate: Audio sample rate, defaults to 44100
    :type sampleRate: int, optional
    :param frameSizeSamples: frame size in audio samples, defaults to 2048
    :type frameSizeSamples: int, optional
    :param hopSizeSamples: hop size in audio samples, defaults to 512
    :type hopSizeSamples: int, optional
    """

    def __init__(self, numMFCCs=20, **kwargs):
        """
        Contructor
        """

        self.numMFCCs = numMFCCs
        super().__init__(numMFCCs, **kwargs)


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

        if audio.getSampleRate() != self.sampleRate:
            raise ValueError(
                'audio buffer samplerate does not equal feature '
                'extraction rate, %s != %s' % (audio.getSampleRate(), self.sampleRate)
            )

        features = librosa.feature.mfcc(
            y=audio.getAudio(),
            sr=self.sampleRate,
            n_fft=self.frameSize,
            hop_length=self.hopSize,
            n_mfcc=self.numMFCCs
        )

        if normalize:
            features = self.normalize(features)

        if self.timeMajor:
            features = np.transpose(features)

        return features
