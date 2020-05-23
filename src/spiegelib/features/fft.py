#!/usr/bin/env python
"""
Fast Fourier Transform (FFT)
"""

import numpy as np

from spiegelib import AudioBuffer
from spiegelib.features.features_base import FeaturesBase
import spiegelib.core.utils as utils

class FFT(FeaturesBase):
    """
    Args:
        fft_sze (int, optional): Size of FFT to use. If set, will truncate input if
            input is smaller than FFT size. If FFT size is larger than input, will zero-pad.
            Defaults to None, so FFT will be the size of input.
        output (str, optional): output type, must be one of ['complex', 'magnitude',
            'power', 'magnitude_phase', 'power_phase'] Defaults to 'complex'.
        scale_axis (int, tuple, None): When applying scaling, determines which dimensions
            scaling be applied along. Defaults to None, which will flatten results and
            calculate scaling variables on that.
        kwargs: Keyword arguments, see :class:`spiegelib.features.features_base.FeaturesBase`.
    """

    def __init__(self, fft_size=None, output='complex', scale_axis=None, **kwargs):
        """
        Contructor
        """

        # Setup feature base class -- FFT is time summarized, so no
        # time slices are used, defaults to normalizing the entire result
        # as opposed to normalizing across each bin separately.
        super().__init__(scale_axis=scale_axis, **kwargs)

        self.fft_size = fft_size

        if output not in utils.spectrum_types:
            raise TypeError('output must be one of %s' % utils.spectrum_types)

        self.output = output
        self.dtype = np.float32
        self.complex_dtype = np.complex64


    def get_features(self, audio):
        """
        Run FFT on audio buffer.

        Args:
            audio (:ref:`AudioBuffer <audio_buffer>`): input audio

        Returns:
            np.ndarray: Results of FFT. Format depends on output type set during\
                construction.
        """

        if not isinstance(audio, AudioBuffer):
            raise TypeError('audio must be AudioBuffer, recieved %s' % type(audio))

        # Determine the length of audio input and determine number of FFT
        # samples to keep.
        buffer = audio.get_audio()
        n_samples = len(buffer)
        n_output = int((n_samples/2) + 1)

        # Run Fast Fourier Transform
        spectrum = np.fft.fft(audio.get_audio(), n=self.fft_size)[0:n_output]
        features = utils.convert_spectrum(spectrum, self.output, dtype=self.dtype,
                                          complex_dtype=self.complex_dtype)

        return features
