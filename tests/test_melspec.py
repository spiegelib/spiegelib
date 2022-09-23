"""
Tests for MelSpectrogram class

TODO -- include a test that verifies the output
"""
import pytest

import numpy as np

from spiegelib import AudioBuffer
import spiegelib.features as sf
import torch
import torchaudio

import utils

class TestMelspectrogram():

    def test_empty_construction(self):
        mel = sf.MelSpectrogram()


    def test_librosa_equal_to_torch(self):

        sample_rate = 44100
        n_fft = 2048
        win_length = None
        hop_length = 512
        center =True
        pad_mode ="reflect"
        power = 2.0
        norm ='slaney'
        onesided = True
        n_mels = 128

        sine = utils.make_test_sine(2048, 440, sample_rate)
        audio = AudioBuffer(sine, sample_rate)
        mel = sf.MelSpectrogram(sample_rate=sample_rate,
                             frame_size=n_fft,
                             window_length=win_length,
                             hop_size=hop_length,
                             center=True,
                             pad_mode="reflect",
                             power=2.0,
                             norm='slaney',
                             n_mels=n_mels)

        librosa_mel = mel(audio)

        torch_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            onesided=True,
            n_mels=n_mels
        )(torch.Tensor(sine))

        mse = ((torch_mel - librosa_mel) ** 2).mean()
        error = 0.000001
        assert mse < error

