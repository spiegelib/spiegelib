#!/usr/bin/env python
"""
Abstract Base Class for Audio Features
"""

from abc import ABC, abstractmethod
import numpy as np


class FeaturesBase(ABC):
    """
    :param sampleRate: Audio sample rate, defaults to 44100
    :type sampleRate: int, optional
    :param frameSizeSamples: frame size in audio samples, defaults to 2048
    :type frameSizeSamples: int, optional
    :param hopSizeSamples: hop size in audio samples, defaults to 512
    :type hopSizeSamples: int, optional
    """

    def __init__(self, sampleRate=44100, frameSizeSamples=2048, hopSizeSamples=512):
        """
        Constructor
        """

        super().__init__()

        self.sampleRate = sampleRate
        self.frameSizeSamples = frameSizeSamples
        self.hopSizeSamples = hopSizeSamples


    @abstractmethod
    def getFeatures(self, audio, normalize=True):
        """
        Must be implemented. Run audio feature extraction on audio provided as parameter.
        Normalization should be applied based on the normalize parameter.

        :param audio: Audio to process features on
        :type audio: np.array
        :param normalize: Whether or not the features are normalized, defaults to True
        :type normalize: bool, optional
        :returns: results from audio feature extraction
        :rtype: np.array
        """
        pass
