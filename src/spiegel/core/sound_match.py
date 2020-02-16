#!/usr/bin/env python
"""
Class for Automatic Sound Matching
"""

import os
import librosa
from spiegel.synth.synth_base import SynthBase
from spiegel.features.features_base import FeaturesBase, NormalizerError
from spiegel.estimator.estimator_base import EstimatorBase

class SoundMatch():
    """
    :param synth: synthesizer, must inherit from :class:`spiegel.synth.synth_base.SynthBase`.
    :type synth: Object
    :param features: feature extraction, must inherit from :class:`spiegel.features.feature_base.FeatureBase`.
    :type features: Object
    :param estimator: paramter estimator, must inherit from :class:`spiegel.estimator.estimator_base.EstimatorBase`.
    :type estimator: Object
    """

    def __init__(self, synth, features, estimator):
        """
        Constructor
        """

        # Check for valid synth
        if isinstance(synth, SynthBase):
            self.synth = synth
        else:
            raise TypeError('synth must inherit from SynthBase, received %s' % type(synth))

        # Check for valid feature extraction object
        if isinstance(features, FeaturesBase):
            self.features = features
        else:
            raise TypeError('features must inherit from Featurebase, received %s' % type(features))

        # Check for valid estimator
        if isinstance(estimator, EstimatorBase):
            self.estimator = estimator
        else:
            raise TypeError('estimator must inherit from EstimatorBase, received type %s' % type(estimator))


    def match(self, target):
        """
        Attempt to estimate parameters for target audio

        :param target: input audio to use as target
        :type target: np.ndarray
        """

        # Attempt to run feature extraction with normalization first
        try:
            features = self.features.getFeatures(target, normalize=True)
        except NormalizerError:
            features = self.features.getFeatures(target)

        if features.shape != self.estimator.inputShape:
            raise Exception('Features shape does not match estimator input shape. %s != %s' % (
                features.shape, self.estimator.inputShape
            ))

        # Estimate parameters
        params = self.estimator.predict(features)

        # Load patch into synth and return audio
        self.synth.setPatch(params)
        self.synth.renderPatch()
        return self.synth.getAudio()


    def matchFromFile(self, audioPath):
        """
        Load audio file from disk and perform sound matching on it

        :param audioPath: filepath
        :type audioPath: str
        :returns: resulting audio from sound matching
        :rtype: np.ndarray
        """

        target, _ = librosa.core.load(audioPath, sr=self.features.sampleRate)
        return self.match(target)
