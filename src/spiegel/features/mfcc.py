#!/usr/bin/env python
"""
MFCC Audio Feature Extractor
"""

import librosa
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

    def __init__(self, numMFCCs=20, sampleRate=44100, frameSizeSamples=2048, hopSizeSamples=512):
        """
        Contructor
        """

        self.numMFCCs = numMFCCs
        super().__init__(
            self.numMFCCs,
            sampleRate=sampleRate,
            frameSizeSamples=frameSizeSamples,
            hopSizeSamples=hopSizeSamples
        )



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

        features = librosa.feature.mfcc(
            y=audio,
            sr=self.sampleRate,
            n_fft=self.frameSizeSamples,
            hop_length=self.hopSizeSamples,
            n_mfcc=self.numMFCCs
        )

        if normalize:
            features = self.normalize(features)

        return features
