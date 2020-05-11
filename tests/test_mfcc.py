"""
Tests for MFCC class

TODO -- include a test that verifies the output
"""

import pytest
import numpy as np

from spiegelib import AudioBuffer
from spiegelib.features import MFCC

import utils

class TestMFCC():

    def test_empty_construction(self):
        mfcc = MFCC()


    def test_sine_mfcc(self):

        sine = utils.make_test_sine(2048, 440, 44100)
        audio = AudioBuffer(sine, 44100)
        mfcc = MFCC(hop_size=512, frame_size=1024)
        features = mfcc(audio)

        assert features.shape == (20,5)


    def test_sine_mfcc_time_major(self):

        sine = utils.make_test_sine(2048, 440, 44100)
        audio = AudioBuffer(sine, 44100)
        mfcc = MFCC(hop_size=512, frame_size=1024, time_major=True)
        features = mfcc(audio)

        assert features.shape == (5,20)


    def test_sine_mfcc_scale(self):

        batch_size = 10
        mfcc = MFCC(hop_size=512, frame_size=1024, scale_axis=(0,2))
        features = np.zeros((batch_size, 20, 5))

        for i in range(batch_size):
            sine = utils.make_test_sine(2048, 100 + (50 * i), 44100)
            audio = AudioBuffer(sine, 44100)
            features[i] = mfcc(audio)

        assert features.shape == (10, 20,5)
        scaled = mfcc.fit_scaler(features)
        assert scaled.mean() == pytest.approx(0.)
        assert scaled.std() == pytest.approx(1.)
        np.testing.assert_array_almost_equal(scaled.mean((0,2)), np.zeros(20))
        np.testing.assert_array_almost_equal(scaled.std((0,2)), np.ones(20))


    def test_sine_mfcc_scale_time_major(self):

        batch_size = 10
        mfcc = MFCC(13, hop_size=512, frame_size=1024, time_major=True, scale_axis=(0,1))
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
