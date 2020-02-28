#!/usr/bin/env python
"""
Dataset Generator Class
"""

import os
import numpy as np
from tqdm import trange
import scipy.io.wavfile

from spiegel.synth.synth_base import SynthBase
from spiegel.features.features_base import FeaturesBase
from spiegel.features.mfcc import MFCC


class DatasetGenerator():
    """
    :param synth: Synthesizer to generate test data from. Must inherit from
        :class:`spiegel.synth.synth_base.SynthBase`.
    :type synth: Object
    :param features: Features to use for dataset generation. Must inherit from
        :class:`spiegel.features.features_base.FeaturesBase`
    :type features: Object
    :param output_folder: Output folder for dataset, defaults to currect working directory
    :type output_folder: str, optional
    :param save_audio: whether or not to save rendered audio files, defaults to False
    :type save_audio: bool, optional
    :param normalize: whether or not to normalize features. Requires the normalizers in the
        feature object to be pre-trained. :py:meth:`generate` can be used to train the normalizers.
        Defaults to True.
    :type normalize: bool, optional

    :cvar features_filename: filename for features output file, defaults to features.npy
    :vartype features_filename: str
    :cvar patches_filename: filename for patches output file, defaults to patches.npy
    :vartype patches_filename: str
    :cvar audio_folder_name: folder name for the audio output if used. Will be automatically
        created within the output folder if saving audio. Defaults to audio
    :vartype audio_folder_name: str
    """

    def __init__(self, synth, features, output_folder=os.getcwd(), save_audio=False, normalize=True):
        """
        Contructor
        """

        # Check for valid synth
        if isinstance(synth, SynthBase):
            self.synth = synth
        else:
            raise TypeError('synth must inherit from SynthBase')

        # Check for valid features
        if isinstance(features, FeaturesBase):
            self.features = features
        else:
            raise TypeError('features must inherit from FeaturesBase')

        # Check for valid output folder
        self.output_folder = os.path.abspath(output_folder)
        if not (os.path.exists(self.output_folder) and os.path.isdir(self.output_folder)):
            os.mkdir(self.output_folder)

        self.save_audio = save_audio

        # Default folder for audio output
        self.audio_folder_name = "audio"

        # Default filenames for output files
        self.features_filename = "features.npy"
        self.patches_filename = "patches.npy"

        # Should the feature set data be normalized?
        self.normalize = normalize


    def generate(self, size, file_prefix="", fit_normalizers_only=False):
        """
        Generate dataset with a set of random patches

        :param size: Number of patches to include in dataset
        :type size: int
        :param file_prefix: filename prefix for output dataset, defaults to ""
        :type file_prefix: str, optional
        """

        # Get a single example to determine required array size required
        audio = self.synth.get_random_example()
        features = self.features(audio)
        patch = self.synth.get_patch()

        # Arrays to hold dataset
        shape = list(features.shape)
        shape.insert(0, size)
        feature_set = np.empty(shape, dtype=features.dtype)
        patch_set = np.zeros((size, len(patch)), dtype=np.float32)

        # Should the features be normalized with the feature normalizers?
        normalize = self.normalize and self.features.has_normalizers()

        # Generate data
        for i in trange(size, desc="Generating Dataset"):
            audio = self.synth.get_random_example()
            feature_set[i] = self.features(audio, normalize=normalize)
            patch_set[i] = [p[1] for p in self.synth.get_patch()]

            # Save rendered audio if required
            if self.save_audio:
                self.create_audio_folder()
                audio.save(os.path.join(self.audio_folder_path, "%soutput_%s.wav" % (file_prefix, i)))

        # If only fitting normalizers, do that and return. Don't save any data
        if fit_normalizers_only:
            print("Fitting normalizers only")
            self.features.fit_normalizers(feature_set, transform=False)
            return

        if self.normalize and not self.features.has_normalizers():
            print("Fitting normalizers and normalizing data")
            feature_set = self.features.fit_normalizers(feature_set)

        # Save dataset
        np.save(os.path.join(self.output_folder, "%s%s" % (file_prefix, self.features_filename)), feature_set)
        np.save(os.path.join(self.output_folder, "%s%s" % (file_prefix, self.patches_filename)), patch_set)


    def create_audio_folder(self):
        """
        Check for and create the audio output folder if necassary
        """
        self.audio_folder_path = os.path.abspath(os.path.join(self.output_folder, self.audio_folder_name))
        if not (os.path.exists(self.audio_folder_path) and os.path.isdir(self.audio_folder_path)):
            os.mkdir(self.audio_folder_path)


    def save_normalizers(self, file_name):
        """
        Save feature normalizers

        :param file_name: file name for normalizer pickle
        :type file_name: str
        """
        self.features.save_normalizers(os.path.join(self.output_folder, file_name))
