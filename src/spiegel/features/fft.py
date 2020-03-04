#!/usr/bin/env python
"""
FFT Audio Feature Extractor
"""

import numpy as np
from spiegel import AudioBuffer
from spiegel.features.features_base import FeaturesBase

class FFT(FeaturesBase):
    """
    :param fft_size: Size of FFT, defaults to None. If set, will truncate input
        if smaller than input size, or zero pad if longer.
    :type fft_size: int, optional
    :param kwargs: keyword arguments for base class, see :class:`spiegel.features.features_base.FeaturesBase`.
    """

    output_types = ['complex', 'magnitude', 'power', 'magnitude_phase', 'power_phase']

    def __init__(self, fft_size=None, output='magnitude', **kwargs):
        """
        Contructor
        """

        super().__init__(0, per_feature_normalize=False, **kwargs)

        self.frame_size = fft_size

        if output not in type(self).output_types:
            raise TypeError('output must be one of %s' % str(type(self).output_types))

        self.output = output

        if output == 'complex':
            self.dtype = np.complex128
        else:
            self.dtype = np.float64


    def get_features(self, audio):
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

        # Determine the length of audio input and determine number of FFT
        # samples to keep.
        buffer = audio.get_audio()
        n_samples = len(buffer)
        n_output = int((n_samples/2) + 1)
        self.dimensions = n_output

        # Run Fast Fourier Transform
        spectrum = np.fft.fft(audio.get_audio(), n=self.frame_size)[0:n_output]

        # Convert to desired output format and data type
        if self.output == 'magnitude':
            features = np.array(np.abs(spectrum), dtype=self.dtype)

        elif self.output == 'power':
            features = np.array(np.abs(spectrum)**2, dtype=self.dtype)

        elif self.output == 'magnitude_phase':
            features = np.empty((n_output, 2), dtype=self.dtype)
            features[:,0] = np.abs(spectrum)
            features[:,1] = np.angle(spectrum)

        elif self.output == 'power_phase':
            features = np.empty((n_output, 2), dtype=self.dtype)
            features[:,0] = np.abs(spectrum)**2
            features[:,1] = np.angle(spectrum)

        else:
            features = np.array(spectrum, dtype=self.dtype)

        return features
