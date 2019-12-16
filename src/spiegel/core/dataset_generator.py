#!/usr/bin/env python
"""
Dataset Generator Class
"""

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
    """

    def __init__(self, synth, features=MFCC()):
        """
        Contructor
        """

        if isinstance(synth, SynthBase):
            self.synth = synth
        else:
            raise TypeError('synth must inherit from SynthBase')

        if isinstance(features, FeaturesBase):
            self.features = features
        else:
            raise TypeError('features must inherit from FeaturesBase')


    def generate():
        """
        Generate dataset
        """
