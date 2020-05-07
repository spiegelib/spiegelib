"""
Tests for STFT class
"""

import pytest
import numpy as np

from spiegelib import AudioBuffer
from spiegelib.features import STFT

import utils

class TestSTFT():

    def test_empty_construction(self):
        stft = STFT()


    def test_sine_wave_stft_complex(self, shared_datadir):
        file_name = (shared_datadir / 'audio/test_sine.wav').resolve()
        audio = AudioBuffer()
        audio.load(file_name)
        stft = STFT()
        features = stft(audio)

        assert features.shape == (513,87)
        assert isinstance(features[0][0], np.complex64)


    def test_sine_wave_stft_complex_time_major(self, shared_datadir):
        file_name = (shared_datadir / 'audio/test_sine.wav').resolve()
        audio = AudioBuffer()
        audio.load(file_name)
        stft = STFT(time_major=True, fft_size=2048, hop_size=1024)
        features = stft(audio)

        assert features.shape == (44,1025)
        assert isinstance(features[0][0], np.complex64)


    def test_sine_wave_stft_magnitude(self):

        bin_freq = 44100. / 1024.
        sine = utils.make_test_cosine(1024 + 1, bin_freq*10, 44100)
        audio = AudioBuffer(sine, 44100)
        stft = STFT(output='magnitude', fft_size=1024)
        features = stft(audio)
        expected = np.zeros((513,3))

        # amplitude is spread out over neighbouring bins
        expected[9,:] = 128.
        expected[10,:] = 256.
        expected[11,:] = 128.

        assert features.shape == (513,3)
        assert isinstance(features[0][0], np.float32)
        np.testing.assert_array_almost_equal(features, expected)


    def test_sine_wave_stft_power(self):

        bin_freq = 44100. / 1024.
        sine = utils.make_test_cosine(1024 + 1, bin_freq*10, 44100)
        audio = AudioBuffer(sine, 44100)
        stft = STFT(output='power', fft_size=1024)
        features = stft(audio)
        expected = np.zeros((513,3))

        # amplitude is spread out over neighbouring bins
        expected[9,:] = 128. * 128.
        expected[10,:] = 256. * 256.
        expected[11,:] = 128. * 128.

        assert features.shape == (513,3)
        assert isinstance(features[0][0], np.float32)
        np.testing.assert_array_almost_equal(features, expected)



    def test_sine_wave_stft_magnitude_phase(self):

        bin_freq = 44100. / 1024.
        sine = utils.make_test_cosine(1024 + 1, bin_freq*10, 44100)
        audio = AudioBuffer(sine, 44100)
        stft = STFT(output='magnitude_phase', fft_size=1024)
        features = stft(audio)
        expected = np.zeros((513,3,2))

        # amplitude is spread out over neighbouring bins
        expected[9,:,0] = 128.
        expected[10,:,0] = 256.
        expected[11,:,0] = 128.

        assert features.shape == (513,3,2)
        assert isinstance(features[0][0][0], np.float32)
        np.testing.assert_array_almost_equal(features[:,:,0], expected[:,:,0])
        np.testing.assert_array_almost_equal(features[10,:,1], expected[10,:,1])


    def test_sine_wave_stft_power_phase(self):

        bin_freq = 44100. / 1024.
        sine = utils.make_test_cosine(1024 + 1, bin_freq*10, 44100)
        audio = AudioBuffer(sine, 44100)
        stft = STFT(output='power_phase', fft_size=1024)
        features = stft(audio)
        expected = np.zeros((513,3,2))

        # amplitude is spread out over neighbouring bins
        expected[9,:,0] = 128. * 128.
        expected[10,:,0] = 256. * 256.
        expected[11,:,0] = 128. * 128.

        assert features.shape == (513,3,2)
        assert isinstance(features[0][0][0], np.float32)
        np.testing.assert_array_almost_equal(features[:,:,0], expected[:,:,0])
        np.testing.assert_array_almost_equal(features[10,:,1], expected[10,:,1])


    def test_sine_wave_stft_magnitude_scaled(self):

        bin_freq = 44100. / 1024.
        length = (1024. * 10) + 1
        stft = STFT(output='magnitude', fft_size=1024, hop_size=512, time_major=True)
        batch_size = 10
        features = np.zeros((10,21,513))

        for i in range(batch_size):
            sine = utils.make_test_cosine(int(length), bin_freq*(i+10), 44100)
            audio = AudioBuffer(sine, 44100)
            features[i] = stft(audio)

        scaled = stft.fit_normalizers(features)
        assert scaled.shape == (10,21,513)
        assert scaled.mean() == pytest.approx(0.)
        assert scaled.std() == pytest.approx(1.)
