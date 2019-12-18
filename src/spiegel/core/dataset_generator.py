#!/usr/bin/env python
"""
Dataset Generator Class
"""

import os
import numpy as np
from tqdm import trange

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
    :param outputFolder: Output folder for dataset, defaults to currect folder
    :type outputFolder: str, optional

    :cvar featuresFileName: filename for features output file, defaults to features.npy
    :vartype featuresFileName: str
    :cvar patchesFileName: filename for patches output file, defaults to patches.npy
    :vartype patachesFileName: str
    """

    def __init__(self, synth, features=MFCC(), outputFolder='./'):
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
            raise TypeError('Output folder must be a valid directory')

        # Default filenames for output files
        self.featuresFileName = "features.npy"
        self.patchesFileName = "patches.npy"


    def generate(self, size, filePrefix=""):
        """
        Generate dataset with a set of random patches

        :param size: Number of patches to include in dataset
        :type size: int
        """

        # Get a single example to determine required array size required
        audio = self.synth.getRandomExample()
        features = self.features.getFeatures(audio)
        patch = self.synth.getPatch()

        # Arrays to hold dataset
        featureSet = np.zeros((size, features.shape[0], features.shape[1]), dtype=np.float32)
        patchSet = np.zeros((size, len(patch)), dtype=np.float32)

        # Generate data
        for i in trange(size, desc="Generating Dataset"):
            audio = self.synth.getRandomExample()
            featureSet[i] = self.features.getFeatures(audio)
            patchSet[i] = [p[0] for p in self.synth.getPatch()]

        # Save dataset
        np.save("%s/%s%s" % (self.outputFolder, filePrefix, self.featuresFileName), featureSet)
        np.save("%s/%s%s" % (self.outputFolder, filePrefix, self.patchesFileName), patchSet)
