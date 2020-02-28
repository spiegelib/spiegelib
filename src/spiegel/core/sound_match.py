#!/usr/bin/env python
"""
Class for Automatic Sound Matching
"""

import os
import librosa
from spiegel import AudioBuffer
from spiegel.synth.synth_base import SynthBase
from spiegel.features.features_base import FeaturesBase, NormalizerError
from spiegel.estimator.estimator_base import EstimatorBase

class SoundMatch():
    """
    :param synth: synthesizer, must inherit from
        :class:`spiegel.synth.synth_base.SynthBase`.
    :type synth: Object
    :param features: feature extraction, must inherit from
        :class:`spiegel.features.feature_base.FeatureBase`.
    :type features: Object
    :param estimator: paramter estimator, must inherit from
        :class:`spiegel.estimator.estimator_base.EstimatorBase`.
    :type estimator: Object
    """

    def __init__(self, synth, estimator, features=None):
        """
        Constructor
        """

        # Check for valid synth
        if isinstance(synth, SynthBase):
            self.synth = synth
        else:
            raise TypeError('synth must inherit from SynthBase, received %s' % type(synth))

        # Check for valid estimator
        if isinstance(estimator, EstimatorBase):
            self.estimator = estimator
        else:
            raise TypeError('estimator must inherit from EstimatorBase, received type %s' % type(estimator))

        # Check for valid feature extraction object
        self.features = None
        if features:
            if isinstance(features, FeaturesBase):
                self.features = features
            else:
                raise TypeError('features must inherit from Featurebase, received %s' % type(features))

        self.patch = None


    def get_patch(self):
        """
        Return the estimated sound matched patch
        """
        if not self.patch:
            raise Exception('Please run match first')

        return self.patch


    def match(self, target):
        """
        Attempt to estimate parameters for target audio

        :param target: input audio to use as target
        :type target: :class:`spiegel.core.audio_buffer.AudioBuffer`
        """

        # Attempt to run feature extraction if features have been provided
        if self.features:
            # Try normalization first
            try:
                print('here normalizing')
                input = self.features(target, normalize=True)
            except NormalizerError:
                input = self.features(target)
        else:
            input = target

        # Estimate parameters
        params = self.estimator.predict(input)

        # Load patch into synth and return audio
        self.synth.set_patch(params)
        self.synth.render_patch()
        self.patch = self.synth.get_patch()
        return self.synth.get_audio()


    def match_from_file(self, path):
        """
        Load audio file from disk and perform sound matching on it

        :param path: filepath
        :type path: str
        :returns: resulting audio from sound matching
        :rtype: np.ndarray
        """

        target = AudioBuffer(path, self.features.sample_rate)
        return self.match(target)
