#!/usr/bin/env python
"""
Class for handling audio signals
"""

import numbers
import numpy as np
import librosa
import scipy.io.wavfile


class AudioBuffer():
    """
    :param input: Can be an array of audio samples (np.ndarray or list), or a path
        to a location of an audio file on disk. Defaults to None.
    :type input: optional, np.ndarray, list, str, file-like object
    :param sampleRate: rate of sampled audio if audio data was passed in, or rate
        to resample audio data loaded from disk at, defaults to None.
    :type sampleRate: optional, int
    """

    def __init__(self, input=None, sampleRate=None):
        """
        Constructor
        """

        self.audioData = None
        self.sampleRate = None

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

        audioSet = isinstance(audio, np.ndarray)

        # If an audio file location was provided, load audio from disk on construction
        if path:
            self.load(path, sampleRate=(sampleRate if sampleRate else 44100))

        # If audio data was passed in directly
        elif audioSet and sampleRate:
            self.replaceAudioData(audio, sampleRate)

        # raise exception if auio data passed in but not sampleRate
        elif audioSet and not sampleRate:
            raise Exception('Sample rate is required when initializing with audio data')


    def getAudio(self):
        """
        Getter for audio data

        :returns: Array of audio samples
        :rtype: np.ndarray
        """
        return self.audioData


    def getSampleRate(self):
        """
        Getter for sample rate

        :returns: Sample rate of audio buffer
        :rtype: int
        """
        return self.sampleRate


    def replaceAudioData(self, audio, sampleRate):
        """
        Replace audio data in this object

        :param audio: array of audio samples
        :type audio: np.ndarray
        :param sampleRate: rate that audio data was sampled at
        :type sampleRate: int
        """

        self.audioData = audio
        self.sampleRate = sampleRate


    def load(self, path, sampleRate=44100, **kwargs):
        """
        Read audio from a file into a numpy array at specific audio rate.
        Uses librosa's audio load function, see ` documentation
        <https://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load>`_.
        for more information.

        :param path: location of audio file on disk
        :type path: str, int, pathlib.Path, or file-like object
        :param sampleRate: resample audio to this rate, defaults to 44100
        :type sampleRate: number
        :param kwargs: keyword args to pass into librosa load function
        :returns: (audio samples, sample rate)
        :rtype: tuple (np.ndarray, number)
        """

        self.audioData, self.sampleRate = librosa.core.load(path, sr=sampleRate, **kwargs)


    def save(self, path, normalize=False):
        """
        Save an audio to disk as WAV file at given sample rate. Can normalize
        before saving as well.

        :param path: location to save audio to
        :type path: str or open file handle
        :param normalize: normalize audio before writing, defaults to false
        :type normalize: boolean
        """

        audio = self.audioData
        if normalize:
            audio = AudioBuffer.peakNormalize(audio)

        scipy.io.wavfile.write(
            path,
            self.sampleRate,
            audio
        )


    @staticmethod
    def peakNormalize(audio):
        """
        Peak normalize audio data
        """

        maxSample = np.max(np.abs(audio))
        minSample = np.min(np.abs(audio))

        if maxSample > minSample:
            audio /= maxSample

        return audio
