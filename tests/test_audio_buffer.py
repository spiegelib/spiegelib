"""
Tests for AudioBuffer class
"""

import pytest
import numpy as np
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from spiegelib import AudioBuffer

class TestAudioBuffer():

    def make_test_sine(self, size, hz, rate=44100):
        samples = np.zeros(size)
        phaseIncrement = (2*np.pi) / (float(rate) / float(hz))
        phase = 0.0
        for i in range(size):
            samples[i] = np.sin(phase)
            phase = phase + phaseIncrement

        return samples

    def test_empty_construction(self):
        audio = AudioBuffer()
        assert audio.get_audio() == None
        assert audio.get_sample_rate() == None
        assert audio.channels == 0
        assert audio.file_name == ''


    def test_array_construction_mono(self):
        samples = np.array([1,2,3,4,5,6,7,8,9,10])
        audio = AudioBuffer(samples, 44100)
        assert np.array_equal(audio.get_audio(), samples)
        assert audio.get_sample_rate() == 44100
        assert audio.channels == 1
        assert audio.file_name == ''


    def test_list_construction_mono(self):
        samples = [1,2,3,4,5,6,7,8,9,10]
        audio = AudioBuffer(samples, 44100)
        assert np.array_equal(audio.get_audio(), samples)
        assert audio.get_sample_rate() == 44100
        assert audio.channels == 1
        assert audio.file_name == ''


    def test_array_construction_stereo(self):
        samples = np.array([[1,2,3,4,5],[6,7,8,9,10]])
        audio = AudioBuffer(samples, 44100)
        assert np.array_equal(audio.get_audio(), samples)
        assert audio.get_sample_rate() == 44100
        assert audio.channels == 2
        assert audio.file_name == ''


    def test_list_construction_stereo(self):
        samples = [[1,2,3,4,5],[6,7,8,9,10]]
        audio = AudioBuffer(samples, 44100)
        assert np.array_equal(audio.get_audio(), samples)
        assert audio.get_sample_rate() == 44100
        assert audio.channels == 2
        assert audio.file_name == ''


    def test_array_construction_sample_rate_exception(self):
        samples = np.array([1,2,3,4,5,6,7,8,9,10])
        with pytest.raises(Exception) as exc_info:
            audio = AudioBuffer(samples)
        assert exc_info.type is Exception
        assert exc_info.value.args[0] == 'Sample rate is required when initializing with audio data'


    def test_mono_audio_file_construction(self, shared_datadir):
        file_name = (shared_datadir / 'audio/test_sine.wav').resolve()
        audio = AudioBuffer(file_name)
        assert audio.get_audio().shape == (44100,)
        assert audio.get_sample_rate() == 44100
        assert audio.channels == 1
        assert audio.file_name == file_name


    def test_mono_audio_file_load(self, shared_datadir):
        file_name = (shared_datadir / 'audio/test_sine.wav').resolve()
        audio = AudioBuffer()
        audio.load(file_name)
        assert audio.get_audio().shape == (44100,)
        assert audio.get_sample_rate() == 44100
        assert audio.channels == 1
        assert audio.file_name == file_name


    def test_audio_file_load_resample(self, shared_datadir):
        file_name = (shared_datadir / 'audio/test_sine.wav').resolve()
        audio = AudioBuffer()
        audio.load(file_name, 88200)
        assert audio.get_audio().shape == (88200,)
        assert audio.get_sample_rate() == 88200
        assert audio.channels == 1
        assert audio.file_name == file_name


    def test_stereo_audio_file_load(self, shared_datadir):
        file_name = (shared_datadir / 'audio/test_sine_stereo.wav').resolve()
        audio = AudioBuffer()
        audio.load(file_name, mono=False)
        assert audio.get_audio().shape == (2, 44100)
        assert audio.get_sample_rate() == 44100
        assert audio.channels == 2
        assert audio.file_name == file_name


    def test_replace_audio(self):
        samples = np.array([[1,2,3,4,5],[6,7,8,9,10]])
        audio = AudioBuffer(samples, 44100)

        samplesB = np.array([10,9,8,7,6,5,4,3,2,1])
        audio.replace_audio_data(samplesB, 96000)
        assert np.array_equal(audio.get_audio(), samplesB)
        assert audio.channels == 1


    def test_resize_smaller_mono(self):
        samples = np.array([1,2,3,4,5,6,7,8,9,10])
        audio = AudioBuffer(samples, 44100)
        audio.resize(5)
        assert np.array_equal(audio.get_audio(), np.array([1,2,3,4,5]))


    def test_resize_smaller_shift_mono(self):
        samples = np.array([1,2,3,4,5,6,7,8,9,10])
        audio = AudioBuffer(samples, 44100)
        audio.resize(5, 5)
        assert np.array_equal(audio.get_audio(), np.array([6,7,8,9,10]))


    def test_resize_smaller_shift_pad_mono(self):
        samples = np.array([1,2,3,4,5,6,7,8,9,10])
        audio = AudioBuffer(samples, 44100)
        audio.resize(5, 7)
        assert np.array_equal(audio.get_audio(), np.array([8,9,10,0,0]))


    def test_resize_smaller_stereo(self):
        samples = np.array([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]])
        audio = AudioBuffer(samples, 44100)
        audio.resize(5)
        assert np.array_equal(audio.get_audio(), np.array([[1,2,3,4,5],[1,2,3,4,5]]))


    def test_resize_smaller_shift_stereo(self):
        samples = np.array([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]])
        audio = AudioBuffer(samples, 44100)
        audio.resize(5, 5)
        assert np.array_equal(audio.get_audio(), np.array([[6,7,8,9,10],[6,7,8,9,10]]))


    def test_resize_smaller_shift_pad_stereo(self):
        samples = np.array([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]])
        audio = AudioBuffer(samples, 44100)
        audio.resize(5, 7)
        assert np.array_equal(audio.get_audio(), np.array([[8,9,10,0,0],[8,9,10,0,0]]))


    def test_resize_larger_mono(self):
        samples = np.array([1,2,3,4,5])
        audio = AudioBuffer(samples, 44100)
        audio.resize(10)
        assert np.array_equal(audio.get_audio(), np.array([1,2,3,4,5,0,0,0,0,0]))

    def test_resize_larger_stereo(self):
        samples = np.array([[1,2,3,4,5],[1,2,3,4,5]])
        audio = AudioBuffer(samples, 44100)
        audio.resize(10)
        assert np.array_equal(audio.get_audio(), np.array([[1,2,3,4,5,0,0,0,0,0],
                                                           [1,2,3,4,5,0,0,0,0,0]]))

    def test_save_mono(self, tmp_path):
        samples = self.make_test_sine(44100, 100, 44100)
        audio = AudioBuffer(samples, 44100)
        file_name = (tmp_path / 'test_save.wav').resolve()
        audio.save(file_name)
        audioReload = AudioBuffer(file_name)
        np.testing.assert_array_almost_equal(audio.get_audio(), audioReload.get_audio())
        assert audioReload.get_sample_rate() == 44100
        assert audioReload.channels == 1
        assert audioReload.file_name == file_name


    def test_save_mono_create_dir(self, tmp_path):
        samples = self.make_test_sine(44100, 440, 44100)
        audio = AudioBuffer(samples, 44100)
        file_name = (tmp_path / 'temp_dir/test_save.wav').resolve()
        audio.save(file_name)
        audioReload = AudioBuffer(file_name)
        np.testing.assert_array_almost_equal(audio.get_audio(), audioReload.get_audio())
        assert audioReload.get_sample_rate() == 44100
        assert audioReload.channels == 1
        assert audioReload.file_name == file_name


    def test_save_stereo(self, tmp_path):
        # Create a stereo AudioBuffer
        samplesL = self.make_test_sine(44100, 100, 44100)
        samplesR = self.make_test_sine(44100, 200, 44100)
        samples = np.array([samplesL, samplesR])
        audio = AudioBuffer(samples, 44100)

        # Save Audio Buffer
        file_name = (tmp_path / 'test_save.wav').resolve()
        audio.save(file_name)

        # Reload stereo audio file
        audioReload = AudioBuffer()
        audioReload.load(file_name, mono=False)

        # Make sure it looks correct
        np.testing.assert_array_almost_equal(audio.get_audio(), audioReload.get_audio())
        assert audioReload.get_audio().shape == (2,44100)
        assert audioReload.get_sample_rate() == 44100
        assert audioReload.channels == 2
        assert audioReload.file_name == file_name


    def test_save_mono_normalize(self, tmp_path):
        samples = self.make_test_sine(44100, 400, 44100)
        samplesHalfAmp = samples * 0.5

        audio = AudioBuffer(samples, 44100)
        audioNormed = AudioBuffer(samples, 44100)

        file_name = (tmp_path / 'test_save.wav').resolve()
        audio.save(file_name, normalize=True)
        audioReload = AudioBuffer(file_name)

        # Make sure it looks correct
        np.testing.assert_array_almost_equal(audioNormed.get_audio(), audioReload.get_audio(), decimal=5)
        assert audioReload.get_audio().shape == (44100,)
        assert audioReload.get_sample_rate() == 44100
        assert audioReload.channels == 1
        assert audioReload.file_name == file_name


    def test_peak_normalize(self):
        samples = self.make_test_sine(44100, 100)
        samplesHalfAmp = samples * 0.5
        samplesNorm = AudioBuffer.peak_normalize(samples)
        np.testing.assert_array_equal(samples, samplesNorm)


    def test_peak_normalize_no_change(self):
        samples = self.make_test_sine(44100, 100)
        samplesNorm = AudioBuffer.peak_normalize(samples)
        np.testing.assert_array_almost_equal(samples, samplesNorm, decimal=5)


    @pytest.mark.mpl_image_compare(baseline_dir='mpl_baseline',
                                   filename='test_plot_spectrogram.png')
    def test_plot_spectrogram(self):
        samples = self.make_test_sine(44100, 150, 44100)
        audio = AudioBuffer(samples, 44100)

        fig = plt.figure()
        ax = audio.plot_spectrogram()
        plt.plot()
        return fig


    def test_load_folder(self, shared_datadir):
        file_name = (shared_datadir / 'audio/test_folder').resolve()
        audio = AudioBuffer.load_folder(file_name)

        assert len(audio) == 3
        assert audio[0].get_audio().shape == (44100,)
        assert audio[0].get_sample_rate() == 44100
        assert audio[0].file_name == 'test_1.wav'
        assert audio[1].get_audio().shape == (44100,)
        assert audio[1].get_sample_rate() == 44100
        assert audio[1].file_name == 'test_2.wav'
        assert audio[2].get_audio().shape == (44100,)
        assert audio[2].get_sample_rate() == 44100
        assert audio[2].file_name == 'test_3.wav'


    def test_load_folder_resample(self, shared_datadir):
        file_name = (shared_datadir / 'audio/test_folder').resolve()
        audio = AudioBuffer.load_folder(file_name, sample_rate=22050)

        assert len(audio) == 3
        assert audio[0].get_audio().shape == (22050,)
        assert audio[0].get_sample_rate() == 22050
        assert audio[0].file_name == 'test_1.wav'
        assert audio[1].get_audio().shape == (22050,)
        assert audio[1].get_sample_rate() == 22050
        assert audio[1].file_name == 'test_2.wav'
        assert audio[2].get_audio().shape == (22050,)
        assert audio[2].get_sample_rate() == 22050
        assert audio[2].file_name == 'test_3.wav'


    def test_load_folder_with_non_audio_mono(self, shared_datadir):
        file_name = (shared_datadir / 'audio').resolve()
        audio = AudioBuffer.load_folder(file_name)

        assert len(audio) == 2
        assert audio[0].get_audio().shape == (44100,)
        assert audio[0].get_sample_rate() == 44100
        assert audio[0].file_name == 'test_sine.wav'
        assert audio[1].get_audio().shape == (44100,)
        assert audio[1].get_sample_rate() == 44100
        assert audio[1].file_name == 'test_sine_stereo.wav'


    def test_load_folder_with_non_audio_stereo(self, shared_datadir):
        file_name = (shared_datadir / 'audio').resolve()
        audio = AudioBuffer.load_folder(file_name, mono=False)

        assert len(audio) == 2
        assert audio[0].get_audio().shape == (44100,)
        assert audio[0].get_sample_rate() == 44100
        assert audio[0].file_name == 'test_sine.wav'
        assert audio[1].get_audio().shape == (2,44100)
        assert audio[1].get_sample_rate() == 44100
        assert audio[1].file_name == 'test_sine_stereo.wav'


    def test_load_folder_exception(self, shared_datadir):
        file_name = (shared_datadir / 'audio/non_existent').resolve()
        with pytest.raises(Exception) as exc_info:
            audio = AudioBuffer.load_folder(file_name)
        assert exc_info.type is ValueError
        assert exc_info.value.args[0] == '%s is not a directory' % file_name
