#!/usr/bin/env python
"""
Magnitude Spectrum Audio Feature Extractor
"""

import numpy as np
import librosa
from spiegel import AudioBuffer
from spiegel.features.stft import STFT

class MagnitudeSTFT(STFT):
    """
    :param fft_size: number of FFT bins, defaults to 1024 (overrides frame_size)
    :type fft_size: int, optional
    :param kwargs: keyword arguments for base class, see :class:`spiegel.features.features_base.FeaturesBase`.
    """

    def __init__(self, fft_size=1024, **kwargs):
        """
        Contructor
        """

        super().__init__(fft_size, **kwargs)
        self.add_modifier(
            lambda data : librosa.magphase(data)[0],
            'prenormalize'
        )
