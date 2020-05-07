"""
Tests for FFT class
"""

import pytest
import numpy as np

from spiegelib import AudioBuffer
from spiegelib.features import FFT

import utils

class TestFFT():

    def test_empty_construction(self):
        fft = FFT()


    def test_sine_wave_fft_complex(self, shared_datadir):
        file_name = (shared_datadir / 'audio/test_sine.wav').resolve()
        audio = AudioBuffer()
        audio.load(file_name)
        fft = FFT()
        features = fft(audio)

        assert features.shape == (22051,)
        assert isinstance(features[0], np.complex64)


    def test_sine_wave_fft_magnitude(self):

        bin_freq = 44100. / 1024.
        sine = utils.make_test_sine(1024, bin_freq*10, 44100)
        audio = AudioBuffer(sine, 44100)
        fft = FFT(output='magnitude')
        features = fft(audio)
        expected = np.zeros(513)
        expected[10] = 512.

        assert features.shape == (513,)
        assert isinstance(features[0], np.float32)
        np.testing.assert_array_almost_equal(features, expected)


    def test_sine_wave_fft_magnitude_longer(self):

        bin_freq = 44100. / 4096.
        sine = utils.make_test_sine(4096, bin_freq*10, 44100)
        audio = AudioBuffer(sine, 44100)
        fft = FFT(output='magnitude')
        features = fft(audio)
        expected = np.zeros(2049)
        expected[10] = 2048.

        assert features.shape == (2049,)
        assert isinstance(features[0], np.float32)
        np.testing.assert_array_almost_equal(features, expected)


    def test_sine_wave_fft_power(self):

        bin_freq = 44100. / 1024.
        sine = utils.make_test_sine(1024, bin_freq*10, 44100)
        audio = AudioBuffer(sine, 44100)
        fft = FFT(output='power')
        features = fft(audio)
        expected = np.zeros(513)
        expected[10] = 512. * 512.

        assert features.shape == (513,)
        assert isinstance(features[0], np.float32)
        np.testing.assert_array_almost_equal(features, expected)


    def test_sine_wave_fft_magnitude_phase(self):

        bin_freq = 44100. / 1024.

        # Using a cosine here to get zero initial phase
        sine = utils.make_test_cosine(1024, bin_freq*10, 44100)
        audio = AudioBuffer(sine, 44100)
        fft = FFT(output='magnitude_phase')
        features = fft(audio)
        expected = np.zeros((513,2))
        expected[10][0] = 512.

        assert features.shape == (513,2)
        assert isinstance(features[0][0], np.float32)
        assert isinstance(features[0][1], np.float32)

        # Assert correct magnitude
        np.testing.assert_array_almost_equal(features[:,0], expected[:,0])

        # Uncertain about phase, except for 10th bin
        assert features[10][1] == pytest.approx(0.)


    def test_sine_wave_fft_power_phase(self):

        bin_freq = 44100. / 1024.

        # Using a cosine here to get zero initial phase
        sine = utils.make_test_cosine(1024, bin_freq*10, 44100)
        audio = AudioBuffer(sine, 44100)
        fft = FFT(output='power_phase')
        features = fft(audio)
        expected = np.zeros((513,2))
        expected[10][0] = 512. * 512.

        assert features.shape == (513,2)
        assert isinstance(features[0][0], np.float32)
        assert isinstance(features[0][1], np.float32)

        # Assert correct magnitude
        np.testing.assert_array_almost_equal(features[:,0], expected[:,0])

        # Uncertain about phase, except for 10th bin
        assert features[10][1] == pytest.approx(0.)


    def test_batch_scale_magnitude(self):

        bin_freq = 44100. / 1024.
        feature_batch = np.zeros((10, 513))
        fft = FFT(output='magnitude')

        for i in range(10):
            bin = (i+1)*2
            sine = utils.make_test_cosine(1024, bin_freq*bin, 44100)
            audio = AudioBuffer(sine, 44100)
            feature_batch[i] = fft(audio)

        scaled = fft.fit_normalizers(feature_batch)
        expected_mean = (512.0 * 10) / (513.0 * 10)

        assert fft.normalizer.mean == pytest.approx(expected_mean)


    def test_batch_scale_magnitude_phase(self):

        bin_freq = 44100. / 1024.
        feature_batch = np.zeros((10,513,2))
        fft = FFT(output='magnitude_phase')

        for i in range(10):
            bin = (i+1)*2
            sine = utils.make_test_cosine(1024, bin_freq*bin, 44100)
            audio = AudioBuffer(sine, 44100)
            feature_batch[i] = fft(audio)

        scaled = fft.fit_normalizers(feature_batch)
        expected_mean = (512.0 * 10) / (513.0 * 10)

        assert fft.normalizer.mean.shape == (2,)
        assert fft.normalizer.mean[0] == pytest.approx(expected_mean)
        assert scaled.shape == feature_batch.shape
