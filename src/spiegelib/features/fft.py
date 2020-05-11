#!/usr/bin/env python
"""
FFT Audio Feature Extractor
"""

import numpy as np
from spiegelib import AudioBuffer
from spiegelib.features.features_base import FeaturesBase
import spiegelib.core.utils as utils

class FFT(FeaturesBase):
    """
    :param fft_size: Size of FFT, defaults to None. If set, will truncate input
        if smaller than input size, or zero pad if longer.
    :type fft_size: int, optional
    :param kwargs: keyword arguments for base class, see :class:`spiegelib.features.features_base.FeaturesBase`.
    """

    def __init__(self, output='complex', fft_size=None, scale_axis=None, **kwargs):
        """
        Contructor
        """

        # Setup feature base class -- FFT is time summarized, so no
        # time slices are used, defaults to normalizing the entire result
        # as opposed to normalizing across each bin separately.
        super().__init__(scale_axis=scale_axis, **kwargs)

        self.frame_size = fft_size

        if output not in utils.spectrum_types:
            raise TypeError('output must be one of %s' % utils.spectrum_types)

        self.output = output
        self.dtype = np.float32
        self.complex_dtype = np.complex64


    def get_features(self, audio):
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

        # Determine the length of audio input and determine number of FFT
        # samples to keep.
        buffer = audio.get_audio()
        n_samples = len(buffer)
        n_output = int((n_samples/2) + 1)
        self.dimensions = n_output

        # Run Fast Fourier Transform
        spectrum = np.fft.fft(audio.get_audio(), n=self.frame_size)[0:n_output]
        features = utils.convert_spectrum(spectrum, self.output, dtype=self.dtype,
                                         complex_dtype=self.complex_dtype)

        return features
