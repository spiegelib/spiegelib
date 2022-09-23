#!/usr/bin/env python
"""
Mel Spectrograms as defined by Sound2Synth ()
"""

import numpy as np

import librosa
from spiegelib import AudioBuffer
from spiegelib.features.features_base import FeaturesBase
import spiegelib.core.utils as utils

class MelSpectrogram(FeaturesBase):

    """
    Args:
        frame_size (int, optional): Size of FFT to use when calculating MFCCs, defaults to 2048
        window_length (int or None, optional): Specifies the size of the  
        hop_size (int, optional): hop length in samples, defaults to 512
        power (float, optional): Exponent for the magnitude 
        center (boolean, optional): True if signal should be padded 
        norm (str, optional): Normalization mode for triangles in the spectrogram
        pad_mode (str, optional): Padding mode to be used at the edges
        n_mels (int, optional): Number of melbands to generate
        htk (boolean, optional): True if 
        kwargs: Keyword arguments, see :class:`spiegelib.features.features_base.FeaturesBase`.
    """

    def __init__(self,
                 frame_size=2048,
                 window_length=None,
                 hop_size=512,
                 power=2.0,
                 center=True,
                 norm='slaney',
                 pad_mode="reflect",
                 n_mels=128,
                 htk=True,
                 **kwargs):
        
        self.frame_size = frame_size
        self.window_length = None
        self.hop_size = hop_size
        self.power = power
        self.center = center
        self.norm = norm
        self.pad_mode = pad_mode
        self.n_mels = n_mels
        self.htk = htk
        super().__init__(**kwargs)


    def get_features(self, audio):
        """
        Run Melspectrogram extraciton on audio buffer.

        Args:
            audio (:ref:`AudioBuffer <audio_buffer>`): input audio

        Returns:
            np.ndarray: Results of Melspectrogram extraction. Format depends on output type set during\
                construction.
        """

        if not isinstance(audio, AudioBuffer):
            raise TypeError('audio must be AudioBuffer, recieved %s' % type(audio))

        if audio.get_sample_rate() != self.sample_rate:
            raise ValueError(
                'audio buffer samplerate does not equal feature '
                'extraction rate, %s != %s' % (audio.get_sample_rate(), self.sample_rate)
            )

        features = librosa.feature.melspectrogram(
            y=audio.get_audio(),
            sr=self.sample_rate,
            win_length=self.window_length,
            n_fft=self.frame_size,
            hop_length=self.hop_size,
            power=self.power,
            center=self.center,
            norm=self.norm,
            pad_mode=self.pad_mode,
            n_mels=self.n_mels,
            htk=self.htk
        )

        if self.time_major:
            features = np.transpose(features)

        return features
