#!/usr/bin/env python
"""
Mel-Spectrogram
"""

import numpy as np
import librosa
from spiegelib import AudioBuffer
from spiegelib.features.features_base import FeaturesBase


class MelSpectrogram(FeaturesBase):
    """
    Args:
        n_mels (int, optional): number of mels to use
        fft_size (int, optional): size of FFT, defaults to 1024
        hop_size (int, optional): size of hop shift in samples, defuault to 512
        output (str, optional): output type, must be one of ['energy', 'power', 'db']
            Defaults to 'power'.
        scale_axis (int, tuple, None): When applying scaling, determines which dimensions
            scaling be applied along. Defaults to None, which will flatten results and
            calculate scaling variables on that.
        kwargs: Keyword arguments, see :class:`spiegelib.features.features_base.FeaturesBase`.
    """

    def __init__(self, n_mels=128, fft_size=1024, hop_size=512, output='power',
                 scale_axis=None, **kwargs):
        """
        Contructor
        """

        # Output must be one of these types
        assert output in ['power', 'energy', 'db']

        super().__init__(scale_axis=scale_axis, **kwargs)
        self.n_mels = n_mels
        self.frame_size = fft_size
        self.hop_size = hop_size
        self.output = output


    def get_features(self, audio):
        """
        Run MelSpectrogram on audio buffer

        Args:
            audio (:ref:`AudioBuffer <audio_buffer>`): input audio

        Returns:
            np.ndarray: Results of Mel-Spectrogram. Format depends on output type set during\
                construction.
        """

        if not isinstance(audio, AudioBuffer):
            raise TypeError('audio must be AudioBuffer, recieved %s' % type(audio))

        power = 1 if self.output == 'energy' else 2

        features = librosa.feature.melspectrogram(
            y=audio.get_audio(),
            n_fft=self.frame_size,
            hop_length=self.hop_size,
            sr=audio.sample_rate,
            power=power,
            n_mels=self.n_mels
        )

        if self.output == 'db':
            features = librosa.power_to_db(features, ref=np.max)

        if self.time_major:
            features = np.swapaxes(features, 0, 1)

        return features
