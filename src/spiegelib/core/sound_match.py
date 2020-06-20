#!/usr/bin/env python
"""
This class performs synthesizer sound matching by estimating parameters for a synthesizer
in order to match a target input sound. It accepts an implementation of
:ref:`SynthBase <synth_base>` to estimate parameters for and an implementation of
:ref:`EstimatorBase <estimator_base>`, which performs the parameter estimation.
Optionally, feature extraction can be performed on the target audio file prior
to being fed into the estimator using an implementation
of :ref:`FeaturesBase <features_base>`.

Alternatively, instead of a :ref:`SynthBase <synth_base>` object, a synth config
file (JSON synth state, see :py:meth:`~spiegelib.synth.SynthBase.save_state`) can be used.
Sound matching can then be used to return parameters only without requiring a synthesizer.

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

from spiegelib import AudioBuffer
from spiegelib.synth.synth_base import SynthBase
from spiegelib.features.features_base import FeaturesBase
from spiegelib.estimator.estimator_base import EstimatorBase

class SoundMatch():
    """
    Args:
        synth (Object or String): If an object, must inherit from
            :class:`~spiegelib.synth.SynthBase`. If it is a string, it must be the path
            of a synth config JSON file which contains parameter information and the
            list of overridden parameters for a synthesizer.
            see :py:meth:`~spiegelib.synth.SynthBase.save_state`
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

        self.parameters = None
        self.overridden = None

        # Check for valid synth
        if isinstance(synth, SynthBase):
            self.synth = synth

        elif isinstance(synth, str):
            self.synth = None
            self.setup_synth_params(synth)

        else:
            raise TypeError('synth must inherit from SynthBase, received %s' % type(synth))

        # Check for valid estimator
        if isinstance(estimator, EstimatorBase):
            self.estimator = estimator
        else:
            raise TypeError("estimator must inherit from EstimatorBase, "
                            "received type %s" % type(estimator))

        # Check for valid feature extraction object
        self.features = None
        if features:
            if isinstance(features, FeaturesBase):
                self.features = features
            else:
                raise TypeError("features must inherit from Featurebase, "
                                "received %s" % type(features))

        self.patch = None


    def get_patch(self):
        """
        Returns:
            list: The resulting patch after estimation

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

        if self.synth is None:
            raise ValueError("No Synth object. Perhaps you want to run a parameter "
                             "only match? Use match_parameter method instead")

        # Estimate parameters
        params = self.match_parameters(target)

        # Load patch into synth and return audio
        self.synth.set_patch(params)
        self.synth.render_patch()
        self.patch = self.synth.get_patch(skip_overridden=True)
        return self.synth.get_audio()


    def match_parameters(self, target, expand=False):
        """
        Run estimation of parameters and use audio feature extraction if it
        has been set.

        Args:
            target (:ref:`AudioBuffer <audio_buffer>`): input audio to use as target
            expand (bool, optional): If set to True, will take the parameter values
                returned from sound matching algorithm and expand with overridden
                parameters.

        Returns:
            list: estimated parameter values outputed from estimator
        """

        # Attempt to run feature extraction if features have been provided
        if self.features is not None:
            input_data = self.features(target)
        else:
            input_data = target

        # Estimate parameters
        params = self.estimator.predict(input_data)

        if expand and self.parameters is not None and self.overridden is not None:
            param_indices = [p[0] for p in self.parameters]
            params = SynthBase.expand_sub_patch(params, param_indices, self.overridden)
            params = sorted(params + self.overridden, key=lambda p: p[0])
            self.patch = params
            if len(params) != len(self.parameters):
                raise ValueError("Incorrect number of parameters returned. Number of "
                                 "overridden parameters from synth config file + parameters "
                                 "returned from the estimator must equal the full patch for synth")

        elif expand and self.synth is not None:
            self.synth.set_patch(params)
            params = self.synth.get_patch(skip_overridden=False)
            self.patch = params

        elif expand and (self.parameters is None or self.overridden is None):
            raise ValueError("Unable to expand parameters, in order to use this feature please "
                             "load a synthesizer or a synth config JSON file during consruction")

        return params


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


    def setup_synth_params(self, config_location):
        """
        Setup synth param settings from a synth config file. Once this is setup,
        full synthesizer parameter settings can be generated without having to
        use a synth object. Beneficial for running sound matching without linking
        to the VST.

        Args:
            config_location (str): synth config JSON file
        """

        patch, overridden = SynthBase.load_synth_config(config_location)
        self.parameters = sorted(patch + overridden, key=lambda a: a[0])
        self.overridden = overridden
