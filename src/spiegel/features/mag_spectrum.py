#!/usr/bin/env python
"""
Magnitude Spectrum Audio Feature Extractor
"""

import numpy as np
import librosa
from spiegel import AudioBuffer
from spiegel.features.stft import STFT

class MagSpectrum(STFT):
    """
    :param fftSize: number of FFT bins, defaults to 1024 (overrides frameSize)
    :type fftSize: int, optional
    :param kwargs: keyword arguments for base class, see :class:`spiegel.features.features_base.FeaturesBase`.
    """

    def __init__(self, fftSize=1024, **kwargs):
        """
        Contructor
        """

        super().__init__(fftSize, **kwargs)
        self.addModifier(
            lambda data : librosa.magphase(data)[0],
            'prenormalize'
        )
