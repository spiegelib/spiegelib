#!/usr/bin/env python
"""
This class can be used to generate datasets from instances of synthesizers. This is
useful for creating large sets of data for training and validating deep learning models.

Examples
^^^^^^^^

Example generating 50000 training samples and 10000 testing samples from the
*Dexed* VST FM Synthesizer. Each sample is created by creating a random
patch configuration in *Dexed*, and then rendering a one second audio clip of
that patch. A 13-band MFCC is computed on the resulting audio. These audio features
and the synthesizer parameters used to synthesize the audio are saved in numpy files.
Audio features are standardized by removing the mean and scaling to unit variance. The
values used for scaling are saved after the first dataset generation so they
can be used on future data.

.. code-block:: python
    :linenos:

    import spiegelib as spgl

    synth = spgl.synth.SynthVST("/Library/Audio/Plug-Ins/VST/Dexed.vst",
                                note_length_secs=1.0,
                                render_length_secs=1.0)

    # Mel-frequency Cepstral Coefficients audio feature extractor.
    features = spgl.features.MFCC(num_mfccs=13, frame_size=2048,
                                  hop_size=1024 time_major=True)

    # Setup generator for MFCC output and generate 50000 training examples
    # and 10000 testing examples
    generator = spgl.DatasetGenerator(synth, features,
                                      output_folder="./data_FM_mfcc",
                                      normalize=True)
    generator.generate(50000, file_prefix="train_")
    generator.generate(10000, file_prefix="test_")
    generator.save_scaler('data_scaler.pkl')

"""

import os

import numpy as np
import scipy.io.wavfile
from tqdm import trange

from spiegelib.features.features_base import FeaturesBase
from spiegelib.synth.synth_base import SynthBase


class DatasetGenerator():
    """

    Args:
        synth (Object): Synthesizer to generate test data from. Must inherit from
            :class:`spiegelib.synth.SynthBase`.
        features (Object): Features to use for dataset generation. Must inherit from
            :class:`spiegelib.features.FeaturesBase`.
        output_folder (str, optional): Output folder for dataset, defaults to currect working directory.
        save_audio (bool, optional): Whether or not to save rendered audio files, defaults to False.
        scale (bool, optional): Whether or not to scale resulting feature vector. If the feature
            object does not have a scaler set, then this will train a data scaler based on the
            generated dataset and store them in the features object. Call :py:meth:`save_scaler`
            to store scaler settings. Defaults to False.

    Attributes:
        features_filename (str): filename for features output file, defaults to features.npy
        patches_filename (str): filename for patches output file, defaults to patches.npy
        audio_folder_name (str): folder name for the audio output if used. Will be automatically
            created within the output folder if saving audio. Defaults to audio
    """

    def __init__(self, synth, features, output_folder=os.getcwd(), save_audio=False, scale=False):
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

        # Should the feature set data be scaled?
        self.should_scale = scale


    def generate(self, size, file_prefix="", fit_scaler_only=False):
        """
        Generate dataset with a set of random patches. Saves the extracted features
        and parameter settings in separate .npy files. Files are stored in the output
        folder set during construction (defaults to current working directory) and
        saves the features as "features.npy" and patches as "patches.npy". These file
        names can be prefixed with a string set by the file_prefix argument. If audio
        files are being saved (configured during construction), then the audio files
        are saved in a separate audio folder and all audio files are also prefixed
        by the file_prefix.

        Args:
            size (int): Number of different synthesizer patches to render.
            file_prefix (str, optional): filename prefix for all output data.
            fit_scaler_only (bool, optional): If this is set to True, then
                no data will be saved and only scaler will be set or reset
                for the feature object.
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

        # Should the features be normalized with the feature scaler?
        should_scale = self.should_scale and self.features.has_scaler()

        # Generate data
        for i in trange(size, desc="Generating Dataset"):
            audio = self.synth.get_random_example()
            feature_set[i] = self.features(audio, scale=should_scale)
            patch_set[i] = [p[1] for p in self.synth.get_patch()]

            # Save rendered audio if required
            if self.save_audio:
                self._create_audio_folder()
                audio.save(os.path.join(self.audio_folder_path, "%soutput_%s.wav" % (file_prefix, i)))

        # If only fitting scaler, do that and return. Don't save any data
        if fit_scaler_only:
            print("Fitting scaler only", flush=True)
            self.features.fit_scaler(feature_set, transform=False)
            return

        if self.should_scale and not self.features.has_scaler():
            print("Fitting scaler and scaling data", flush=True)
            feature_set = self.features.fit_scaler(feature_set)

        # Save dataset
        np.save(os.path.join(self.output_folder, "%s%s" % (file_prefix, self.features_filename)), feature_set)
        np.save(os.path.join(self.output_folder, "%s%s" % (file_prefix, self.patches_filename)), patch_set)


    def save_scaler(self, file_name):
        """
        Save feature scaler as a pickle file.

        Args:
            file_name (str): file name for scaler pickle file
        """
        self.features.save_scaler(os.path.join(self.output_folder, file_name))


    def _create_audio_folder(self):
        """
        Check for and create the audio output folder if necassary
        """
        self.audio_folder_path = os.path.abspath(os.path.join(self.output_folder, self.audio_folder_name))
        if not (os.path.exists(self.audio_folder_path) and os.path.isdir(self.audio_folder_path)):
            os.mkdir(self.audio_folder_path)
