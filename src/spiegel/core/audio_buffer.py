#!/usr/bin/env python
"""
Class for handling audio signals
"""

import os
import numbers
import numpy as np
import librosa
import scipy.io.wavfile


class AudioBuffer():
    """
    :param input: Can be an array of audio samples (np.ndarray or list), or a path
        to a location of an audio file on disk. Defaults to None.
    :type input: optional, np.ndarray, list, str, file-like object
    :param sample_rate: rate of sampled audio if audio data was passed in, or rate
        to resample audio data loaded from disk at, defaults to None.
    :type sample_rate: optional, int
    """

    def __init__(self, input=None, sample_rate=None):
        """
        Constructor
        """

        self.audio_data = None
        self.sample_rate = None

        path = None
        audio = None

        # If the input is a numpy array or list with numbers then assume
        # we've been passed audio data. Otherwise, assume we've been passed
        # a path to an audio file
        if isinstance(input, np.ndarray):
            audio = input
        elif isinstance(input, list) and isinstance(input[0], numbers.Number):
            audio = np.array(input)
        elif input:
            path = input

        audio_set = isinstance(audio, np.ndarray)

        # If an audio file location was provided, load audio from disk on construction
        if path:
            self.load(path, sample_rate=(sample_rate if sample_rate else 44100))

        # If audio data was passed in directly
        elif audio_set and sample_rate:
            self.replace_audio_data(audio, sample_rate)

        # raise exception if auio data passed in but not sample_rate
        elif audio_set and not sample_rate:
            raise Exception('Sample rate is required when initializing with audio data')


    def get_audio(self):
        """
        Getter for audio data

        :returns: Array of audio samples
        :rtype: np.ndarray
        """
        return self.audio_data


    def get_sample_rate(self):
        """
        Getter for sample rate

        :returns: Sample rate of audio buffer
        :rtype: int
        """
        return self.sample_rate


    def replace_audio_data(self, audio, sample_rate):
        """
        Replace audio data in this object

        :param audio: array of audio samples
        :type audio: np.ndarray
        :param sample_rate: rate that audio data was sampled at
        :type sample_rate: int
        """

        self.audio = audio
        self.sample_rate = sample_rate


    def load(self, path, sample_rate=44100, **kwargs):
        """
        Read audio from a file into a numpy array at specific audio rate.
        Uses librosa's audio load function, see ` documentation
        <https://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load>`_.
        for more information.

        :param path: location of audio file on disk
        :type path: str, int, pathlib.Path, or file-like object
        :param sample_rate: resample audio to this rate, defaults to 44100
        :type sample_rate: number
        :param kwargs: keyword args to pass into librosa load function
        :returns: (audio samples, sample rate)
        :rtype: tuple (np.ndarray, number)
        """

        self.audio, self.sample_rate = librosa.core.load(path, sr=sample_rate, **kwargs)


    def save(self, path, normalize=False):
        """
        Save an audio to disk as WAV file at given sample rate. Can normalize
        before saving as well.

        :param path: location to save audio to
        :type path: str or open file handle
        :param normalize: normalize audio before writing, defaults to false
        :type normalize: boolean
        """

        # Make directory if it doesn't exist
        fullpath = os.path.abspath(path)
        dir = os.path.dirname(fullpath)
        if not os.path.exists(dir):
            os.mkdir(dir)

        audio = self.audio
        if normalize:
            audio = AudioBuffer.peak_normalize(np.copy(audio))

        scipy.io.wavfile.write(
            fullpath,
            self.sample_rate,
            audio
        )


    @staticmethod
    def peak_normalize(audio):
        """
        Peak normalize audio data
        """

        maxSample = np.max(np.abs(audio))
        minSample = np.min(np.abs(audio))

        if maxSample > minSample:
            audio /= maxSample

        return audio


    @staticmethod
    def load_folder(path):
        """
        Try to load a folder of audio samples

        :param path: Path to directory of audio files
        :type path: str
        :returns: list of :class:`spiegel.core.audio_buffer.AudioBuffer`
        :rtype: list
        """

        abspath = os.path.abspath(path)
        if not (os.path.exists(abspath) and os.path.isdir(abspath)):
            raise ValueError('%s is not a directory' % path)

        audio_files = []
        for file in os.scandir(abspath):
            try:
                audioFile = AudioBuffer(os.path.join(abspath, file.name))
                audio_files.append(audioFile)
            except:
                pass

        return audio_files
