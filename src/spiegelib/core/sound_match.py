#!/usr/bin/env python
"""
This class performs synthesizer sound matching by estimating parameters for synthesizer
in order to match a target input sound. It accepts an implementation of :ref:`SynthBase <synth_base>`
to estimate parameters for and an implementation of :ref:`EstimatorBase <estimator_base>`, which
performs the parameter estimation. Optionally, feature extraction can be performed on
the target audio file prior to being fed into the estimator using an implementation
of :ref:`FeaturesBase <features_base>`.

Example
^^^^^^^

Sound matching with a genetic algorithm

.. code-block:: python
    :linenos:

    import spiegelib as spgl

    # Load synth
    synth = spgl.synth.SynthVST("/Library/Audio/Plug-Ins/VST/Dexed.vst")

    # Setup a basic GA using MFCC error as the evaluation function
    ga_extractor = spgl.features.MFCC(num_mfccs=13, hop_size=1024)
    ga = spgl.estimator.BasicGA(synth, ga_extractor)


    # Setup sound match object with synth and GA
    ga_matcher = spgl.SoundMatch(synth, ga)

    # Load a target audio file
    target = spgl.AudioBuffer('./target.wav')

    # Perform sound matching on audio target
    result_audio = ga_matcher.match(target)
    result_patch = ga_matcher.get_patch()

"""

import os

import librosa

from spiegelib import AudioBuffer
from spiegelib.synth.synth_base import SynthBase
from spiegelib.features.features_base import FeaturesBase
from spiegelib.estimator.estimator_base import EstimatorBase

class SoundMatch():
    """
    Args:
        synth (Object): must inherit from :class:`spiegelib.synth.SynthBase`
        estimator (Object): must inherit from :class:`spiegelib.estimator.EstimatorBase`
        features (Object, optional): must inherit from :class:`spiegelib.features.FeatureBase`

    Raises:
        TypeError: If synth, estimator, or features parameters do not inherit from
            the correct base class.
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
        Returns:
            dict: The resuling patch after estimation

        Raises:
            Exception: If sound matching has not been run first
        """
        if not self.patch:
            raise Exception('Please run match first')

        return self.patch


    def match(self, target):
        """
        Attempt to estimate parameters for target audio

        Args:
            target (:ref:`AudioBuffer <audio_buffer>`): input audio to use as target

        Returns:
            :ref:`AudioBuffer <audio_buffer>`: audio output from synthesizer after sound matching
        """

        # Attempt to run feature extraction if features have been provided
        if self.features:
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

        Args:
            filepath (str): location of audio file on disk

        Returns:
            :ref:`AudioBuffer <audio_buffer>`: audio output from synthesizer after sound matching
        """

        target = AudioBuffer(path, self.features.sample_rate)
        return self.match(target)
