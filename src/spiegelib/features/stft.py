#!/usr/bin/env python
"""
Short Time Fourier Transform (STFT)
"""

import numpy as np
import librosa
from spiegelib import AudioBuffer
from spiegelib.features.features_base import FeaturesBase
import spiegelib.core.utils as utils

class STFT(FeaturesBase):
    """
    Args:
        fft_size (int, optional): size of FFT, defaults to 1024
        hop_size (int, optional): size of hop shift in samples, defuault to 512
        output (str, optional): output type, must be one of ['complex', 'magnitude',
            'power', 'magnitude_phase', 'power_phase'] Defaults to 'complex'.
        scale_axis (int, tuple, None): When applying scaling, determines which dimensions
            scaling be applied along. Defaults to None, which will flatten results and
            calculate scaling variables on that.
        kwargs: Keyword arguments, see :class:`spiegelib.features.features_base.FeaturesBase`.
    """

    def __init__(self, fft_size=1024, hop_size=512, output='complex', scale_axis=None, **kwargs):
        """
        Contructor
        """

        super().__init__(scale_axis=scale_axis, **kwargs)

        if output not in utils.spectrum_types:
            raise TypeError('output must be one of %s' % utils.spectrum_types)

        self.output = output
        self.frame_size = fft_size
        self.hop_size = hop_size
        self.dtype = np.float32
        self.complex_dtype = np.complex64


    def get_features(self, audio):
        """
        Run STFT on audio buffer.

        Args:
            audio (:ref:`AudioBuffer <audio_buffer>`): input audio

        Returns:
            np.ndarray: Results of STFT. Format depends on output type set during\
                construction.
        """

        if not isinstance(audio, AudioBuffer):
            raise TypeError('audio must be AudioBuffer, recieved %s' % type(audio))

        features = librosa.stft(
            y=audio.get_audio(),
            n_fft=self.frame_size,
            hop_length=self.hop_size,
        )

        features = utils.convert_spectrum(features, self.output, dtype=self.dtype,
                                          complex_dtype=self.complex_dtype)

        if self.time_major:
            features = np.swapaxes(features, 0, 1)

        return features
