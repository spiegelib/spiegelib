#!/usr/bin/env python
"""
Class for storing and passing audio signals around. Can be instantiated with an
array of audio samples or location to load audio from disk. Stores sample rate,
number of channels, and the file name if loaded from disk.

Also has utility functions for loading and saving audio from disk, normalizing,
resizing buffers, and plotting spectrograms.

Examples
^^^^^^^^

Create a new empty ``AudioBuffer``::

    >>> audio = spiegelib.AudioBuffer()

Create an ``AudioBuffer`` from existing audio samples::

    >>> audio = spiegelib.AudioBuffer(audio_samples, 44100)

Load audio from disk into a new ``AudioBuffer``::

    >>> audio = spiegelib.AudioBuffer('./some_audio_file.wav')

Load an entire folder of audio samples into a list of ``AudioBuffer`` objects::

    >>> audio_files = spiegelib.AudioBuffer.load_folder('./audio_folder')

Plot a spectrogram

.. code-block:: python
    :linenos:

    import spiegelib as spgl
    import matplotlib.pyplot as plt

    audio = spgl.AudioBuffer('./audio_file.wav')
    audio.plot_spectrogram()
    plt.show()

.. image:: ./images/audio_buffer_spect.png

"""

import os
import numbers

import numpy as np
import librosa
import librosa.display
import scipy.io.wavfile

import spiegelib.core.utils as utils


class AudioBuffer():
    """
    Args:
        input (np.ndarray, list, str, file-like object): Can be an array
            of audio samples (np.ndarray or list), or a path to a location of an
            audio file on disk. Defaults to None.
        sample_rate (int): Rate of sampled audio if audio data was passed in, or rate
            to resample audio data loaded from disk at, defaults to None. If loading audio
            from a file, will automatically resample to 44100 if no sample rate provided.
        kwargs: keyword args to pass into `librosa load function <https://librosa.github.io/librosa/generated/librosa.core.load.html>`_.

    """

    def __init__(self, input=None, sample_rate=None, **kwargs):
        """
        Constructor
        """

        self.audio = None
        self.sample_rate = None
        self.channels = 0

        # Will be set if audio is loaded from a file
        self.file_name = ''

        path = None
        audio = None

        # If the input is a numpy array or list with numbers then assume
        # we've been passed audio data. Otherwise, assume we've been passed
        # a path to an audio file
        if isinstance(input, np.ndarray):
            audio = input
        elif isinstance(input, list):
            audio = np.array(input, np.float32)
        elif input:
            path = input

        audio_set = isinstance(audio, np.ndarray)

        # If an audio file location was provided, load audio from disk on construction
        if path:
            self.load(path, sample_rate=(sample_rate if sample_rate else 44100), **kwargs)

        # If audio data was passed in directly
        elif audio_set and sample_rate:
            self.replace_audio_data(audio, sample_rate)

        # raise exception if auio data passed in but not sample_rate
        elif audio_set and not sample_rate:
            raise Exception('Sample rate is required when initializing with audio data')


    def get_audio(self):
        """
        Getter for audio data

        Returns:
            np.ndarray: Array of audio samples
        """
        return self.audio


    def get_sample_rate(self):
        """
        Getter for sample rate

        Returns:
            int: Sample rate of audio buffer
        """
        return self.sample_rate


    def load(self, path, sample_rate=44100, **kwargs):
        """
        Read audio from a file into a numpy array at specific audio rate. Stores
        in this ``AudioBuffer`` object.

        Args:
            path (str, int, pathlib.Path, or file-like object): location of audio file on disk
            sample_rate (int): resample audio to this rate, defaults to 44100. None uses
                native sampling rate.
            kwargs: keyword args to pass into `librosa load function <https://librosa.github.io/librosa/generated/librosa.core.load.html>`_.
        """

        self.audio, self.sample_rate = librosa.core.load(path, sr=sample_rate, **kwargs)
        self.channels = len(self.audio.shape)
        self.file_name = path


    def plot_spectrogram(self, **kwargs):
        """
        Plot spectrogram of this audio buffer. Uses librosas `specshow <https://librosa.github.io/librosa/generated/librosa.display.specshow.html>`_
        function. Uses matplotlib.pyplot to plot spectrogram.

        Defaults to logarithmic y-axis in Hertz and seconds along the x-axis.

        Args:
            kwargs: see `specshow <https://librosa.github.io/librosa/generated/librosa.display.specshow.html>`_.

        Returns:
            axes: The axis handle for figures
        """

        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.get_audio())), ref=np.max)
        y_axis = 'log' if not 'y_axis' in kwargs else kwargs['y_axis']
        x_axis = 's' if not 'y_axis' in kwargs else kwargs['x_axis']
        return librosa.display.specshow(D, y_axis=y_axis, x_axis=x_axis, **kwargs)


    def replace_audio_data(self, audio, sample_rate):
        """
        Replace audio data in this object

        Args:
            audio (np.ndarray): Array of audio samples
            sample_rate (int): Rate that audio data was sampled at
        """

        self.audio = audio
        self.channels = len(self.audio.shape)
        self.sample_rate = sample_rate


    def resize(self, num_samples, start=0):
        """
        Resize audio to a set number of samples. If new length is less than the
        current audio buffer size, then the buffer will be trimmed. If the new
        length is greater than current audio buffer, then the resulting buffer
        will be zero-padded at the end.

        Args:
            num_samples (int): New length of audio buffer in samples
            start_sample (int): Start reading from certain number of samples into
                current buffer
        """

        new_shape = (self.channels, num_samples) if self.channels > 1 else (num_samples,)
        new_audio = np.zeros(new_shape)
        current = self.audio.shape[1] if self.channels > 1 else self.audio.shape[0]
        smaller = min(current-start, num_samples)

        if self.channels > 1:
            new_audio[:,0:smaller] = self.audio[:,start:start+smaller]
        else:
            new_audio[0:smaller] = self.audio[start:start+smaller]

        self.audio = new_audio


    def save(self, path, normalize=False):
        """
        Save an audio to disk as WAV file at given sample rate. Can normalize
        before saving as well.

        Args:
            path (str or open file handle): location to save audio to
            normalize (boolean): normalize audio before writing, defaults to false
        """

        # Make directory if it doesn't exist
        fullpath = os.path.abspath(path)
        dir = os.path.dirname(fullpath)
        if not os.path.exists(dir):
            os.makedirs(dir)

        audio = np.array(self.audio, copy=True, dtype=np.float32)
        if normalize:
            audio = AudioBuffer.peak_normalize(audio)

        if self.channels > 1:
            audio = np.transpose(audio)

        scipy.io.wavfile.write(
            fullpath,
            self.sample_rate,
            audio
        )


    @staticmethod
    def peak_normalize(audio):
        """
        Peak normalize audio data

        Args:
            audio (np.ndarray) : array of audio samples

        Returns:
            np.ndarray : normalized array of audio samples
        """

        maxSample = np.max(np.abs(audio))
        minSample = np.min(np.abs(audio))

        if maxSample > minSample:
            audio /= maxSample

        return audio


    @staticmethod
    def load_folder(path, sort=True, sample_rate=44100, **kwargs):
        """
        Try to load a folder of audio samples

        Args:
            path (str): Path to directory of audio files
            sort (bool): Apply natural sort to file names. Default True
            sample_rate (int): Sample rate to load audio files at. Defualts to 44100
            kwargs: keyword args to pass into `librosa load function <https://librosa.github.io/librosa/generated/librosa.core.load.html>`_.

        Returns:
            list: List of ``AudioBuffer`` objects
        """

        abspath = os.path.abspath(path)
        if not (os.path.exists(abspath) and os.path.isdir(abspath)):
            raise ValueError('%s is not a directory' % path)

        dir = [file for file in os.listdir(abspath) if not file.startswith('.')]
        if sort:
            dir.sort(key=utils.natural_keys)

        audio_files = []
        for file in dir:
            try:
                audioFile = AudioBuffer(os.path.join(abspath, file), sample_rate, **kwargs)
                audioFile.file_name = file
                audio_files.append(audioFile)
            except:
                pass

        return audio_files
