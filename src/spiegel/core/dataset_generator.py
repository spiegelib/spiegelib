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
    :param features: Features to use for dataset generation, defaults to :class:`spiegel.features.mfcc.MFCC`
        Must inherit from :class:`spiegel.features.features_base.FeaturesBase`
    :type features: Object
    :param outputFolder: Output folder for dataset, defaults to currect working directory
    :type outputFolder: str, optional
    :param saveAudio: whether or not to save rendered audio files, defaults to False
    :type saveAudio: bool, optional
    :param normalize: whether or not to normalize features. Requires the normalizers in the
        feature object to be pre-trained. :method:`generate` can be used to train the normalizers.
        Defaults to True.
    :type normalize: bool, optional

    :cvar featuresFileName: filename for features output file, defaults to features.npy
    :vartype featuresFileName: str
    :cvar patchesFileName: filename for patches output file, defaults to patches.npy
    :vartype patchesFileName: str
    :cvar audioFolderName: folder name for the audio output if used. Will be automatically
        created within the output folder if saving audio. Defaults to audio
    :vartype audioFolderName: str
    """

    def __init__(self, synth, features=MFCC(), outputFolder=os.getcwd(), saveAudio=False, normalize=True):
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
        self.outputFolder = os.path.abspath(outputFolder)
        if not (os.path.exists(self.outputFolder) and os.path.isdir(self.outputFolder)):
            os.mkdir(self.outputFolder)

        self.saveAudio = saveAudio

        # Default folder for audio output
        self.audioFolderName = "audio"

        # Default filenames for output files
        self.featuresFileName = "features.npy"
        self.patchesFileName = "patches.npy"

        # Should the feature set data be normalized?
        self.normalize = normalize


    def generate(self, size, filePrefix="", fitNormalizers=False):
        """
        Generate dataset with a set of random patches

        :param size: Number of patches to include in dataset
        :type size: int
        :param filePrefix: filename prefix for output dataset, defaults to ""
        :type filePrefix: str, optional
        :param fitNormalizers: Use this dataset to train/fit the normalizers in the
            feature object. Defaults to False.
        :type fitNormalizers: bool, optional
        """

        # Make sure audio output folder is available if we are saving audio
        if self.saveAudio:
            self.createAudioFolder()

        # Get a single example to determine required array size required
        audio = self.synth.getRandomExample()
        features = self.features.getFeatures(audio)
        patch = self.synth.getPatch()

        # Arrays to hold dataset
        featureSet = np.zeros((size, features.shape[0], features.shape[1]), dtype=np.float32)
        patchSet = np.zeros((size, len(patch)), dtype=np.float32)

        # Should the features be normalized with the feature normalizers?
        normalize = self.normalize and not fitNormalizers

        # Generate data
        for i in trange(size, desc="Generating Dataset"):
            audio = self.synth.getRandomExample()
            featureSet[i] = self.features.getFeatures(audio, normalize=normalize)
            patchSet[i] = [p[1] for p in self.synth.getPatch()]

            # Save rendered audio if required
            if self.saveAudio:
                scipy.io.wavfile.write(
                    os.path.join(self.audioFolderPath, "%soutput_%s.wav" % (filePrefix, i)),
                    self.synth.sampleRate,
                    audio
                )

        # Fit feature normalizers and normalize features if required
        if fitNormalizers:
            results = self.features.fitNormalizers(featureSet, transform=self.normalize)
            if self.normalize:
                featureSet = results

        # Save dataset
        np.save(os.path.join(self.outputFolder, "%s%s" % (filePrefix, self.featuresFileName)), featureSet)
        np.save(os.path.join(self.outputFolder, "%s%s" % (filePrefix, self.patchesFileName)), patchSet)


    def createAudioFolder(self):
        """
        Check for and create the audio output folder if necassary
        """
        self.audioFolderPath = os.path.abspath(os.path.join(self.outputFolder, self.audioFolderName))
        if not (os.path.exists(self.audioFolderPath) and os.path.isdir(self.audioFolderPath)):
            os.mkdir(self.audioFolderPath)


    def saveNormalizers(self, fileName):
        """
        Save feature normalizers

        :param fileName: file name for normalizer pickle
        :type fileName: str
        """
        self.features.saveNormalizers(os.path.join(self.outputFolder, fileName))
