"""
Tests for Spectral Summarized class

"""

import pytest
import numpy as np

from spiegelib import AudioBuffer
from spiegelib.features import SpectralSummarized

import utils

class TestSpectralSummarized():

    def test_empty_construction(self):
        spectral = SpectralSummarized()


    def test_sine_extraction(self):

        frame_size = 1024
        hop_size = 512
        bin_freq = 44100. / float(frame_size)

        sine = utils.make_test_cosine(1024 + 1, bin_freq*10, 44100)
        audio = AudioBuffer(sine, 44100)
        spectral = SpectralSummarized(frame_size=1024, hop_size=512)
        features = spectral(audio)

        assert features.shape == (22,)
        assert features[0] == pytest.approx(bin_freq*10)
        assert features[1] == pytest.approx(0.)
        assert features[2] == pytest.approx(30.452547928239625)
        assert features[3] == pytest.approx(0.)
        assert features[4] == pytest.approx(0.)
        assert features[5] == pytest.approx(0.)
        assert features[6] == pytest.approx(bin_freq*11)
        assert features[7] == pytest.approx(0.)

        # Spectral flatness subbands
        assert features[8] == pytest.approx(44.0823996531185)
        assert features[9] == pytest.approx(0.)
        assert features[10] == pytest.approx(44.0823996531185)
        assert features[11] == pytest.approx(0.)
        assert features[12] == pytest.approx(124.0823996531185)
        assert features[13] == pytest.approx(0.)
        assert features[14] == pytest.approx(44.0823996531185)
        assert features[15] == pytest.approx(0.)
        assert features[16] == pytest.approx(44.0823996531185)
        assert features[17] == pytest.approx(0.)
        assert features[18] == pytest.approx(44.0823996531185)
        assert features[19] == pytest.approx(0.)
        assert features[20] == pytest.approx(44.0823996531185)
        assert features[21] == pytest.approx(0.)


    def test_sine_extraction_scale(self):

        frame_size = 1024
        hop_size = 512
        bin_freq = 44100. / float(frame_size)
        num_frames = 10
        batch_size = 10
        spectral = SpectralSummarized()
        features = np.ndarray((10, 22,))

        for i in range(batch_size):
            sine = utils.make_test_cosine(22050, bin_freq*(i+1), 44100)
            audio = AudioBuffer(sine, 44100)
            features[i] = spectral(audio)

        assert features.shape == (10,22)
        scaled = spectral.fit_normalizers(features)

        assert scaled.mean() == pytest.approx(0.)
        assert scaled.std() == pytest.approx(1.)
        np.testing.assert_array_almost_equal(scaled.mean(0), np.zeros(22))
        np.testing.assert_array_almost_equal(scaled.std(0), np.ones(22))
