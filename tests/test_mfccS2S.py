"""
Tests for MFCCS2S Sound2Synth class

TODO -- include a test that verifies the output
"""

import pytest
import numpy as np

from spiegelib import AudioBuffer
from spiegelib.features import MFCCS2S

import utils

class TestMFCCS2S():

    def test_empty_construction(self):
        mfcc = MFCCS2S()


    def test_sine_mfcc(self):

        sine = utils.make_test_sine(2048, 440, 44100)
        audio = AudioBuffer(sine, 44100)
        mfcc = MFCCS2S(hop_size=512, 
                    frame_size=1024,
                    power=2.0,
                    center=True,
                    pad_mode="reflect",
                    n_mels=128,
                    htk=True)
        features = mfcc(audio)

        assert features.shape == (13,5)


    def test_sine_mfcc_time_major(self):

        sine = utils.make_test_sine(2048, 440, 44100)
        audio = AudioBuffer(sine, 44100)
        mfcc = MFCCS2S(hop_size=512, frame_size=1024, time_major=True)
        features = mfcc(audio)

        assert features.shape == (5,13)


    def test_sine_mfcc_scale(self):

        batch_size = 10
        mfcc = MFCCS2S(hop_size=512, frame_size=1024, scale_axis=(0,2))
        features = np.zeros((batch_size, 13, 5))

        for i in range(batch_size):
            sine = utils.make_test_sine(2048, 100 + (50 * i), 44100)
            audio = AudioBuffer(sine, 44100)
            features[i] = mfcc(audio)

        assert features.shape == (10, 13,5)
        scaled = mfcc.fit_scaler(features)
        assert scaled.mean() == pytest.approx(0.)
        assert scaled.std() == pytest.approx(1.)
        np.testing.assert_array_almost_equal(scaled.mean((0,2)), np.zeros(13))
        np.testing.assert_array_almost_equal(scaled.std((0,2)), np.ones(13))


    def test_sine_mfcc_scale_time_major(self):

        batch_size = 10
        mfcc = MFCCS2S(13, hop_size=512, frame_size=1024, time_major=True, scale_axis=(0,1))
        features = np.zeros((batch_size, 5, 13))

        for i in range(batch_size):
            sine = utils.make_test_sine(2048, 100 + (50 * i), 44100)
            audio = AudioBuffer(sine, 44100)
            features[i] = mfcc(audio)

        assert features.shape == (10,5,13)
        scaled = mfcc.fit_scaler(features)
        assert scaled.mean() == pytest.approx(0.)
        assert scaled.std() == pytest.approx(1.)
        np.testing.assert_array_almost_equal(scaled.mean((0,1)), np.zeros(13))
        np.testing.assert_array_almost_equal(scaled.std((0,1)), np.ones(13))
