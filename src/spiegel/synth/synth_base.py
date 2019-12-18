#!/usr/bin/env python
"""
Synth Abstract Base Class
"""
from __future__ import print_function
from abc import ABC, abstractmethod


class SynthBase(ABC):
    """
    :param sampleRate: sampling rate for rendering audio, defaults to 44100
    :type sampleRate: int, optional
    :param bufferSize: buffer size for rendering audio, defaults to 512
    :type bufferSize: int, optional
    :param midiNote: midi note number used for rendering, 0-127, defaults to 40
    :type midiNote: int, optional
    :param midiVelocity: midi velocity used for rendering. 0-127, defaults to 127
    :type midiVelocity: int, optional
    :param noteLengthSecs: length of midi note in seconds, defaults to 1.0
    :type noteLengthSecs: float, optional
    :param renderLengthSecs: length that audio is rendered for in total, defaults to 2.5
    :type renderLengthSecs: float, optional
    :param overriddenParameters: a list of tuples containing the parameter index to override and the value to lock that parameter to, defaults to []
    :type overriddenParameters: list, optional
    :param normaliseAudio: whether or not to normalize rendered audio, defaults to False
    :type normaliseAudio: boolean, optional

    :cvar renderedPatch: indicates whether a patch has been rendered yet, defaults to False
    :vartype renderedPatch: boolean
    :cvar parameters: parameter indices and names
    :vartype parameters: dict
    :cvar patch: current patch values
    :vartype patch: list
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.sampleRate = kwargs.get('sampleRate', 44100)
        self.bufferSize = kwargs.get('bufferSize', 512)
        self.midiNote = kwargs.get('midiNote', 40)
        self.midiVelocity = kwargs.get('midiVelocity', 127)
        self.noteLengthSecs = kwargs.get('noteLengthSecs', 1.0)
        self.renderLengthSecs = kwargs.get('renderLengthSecs', 2.5)
        self.overriddenParameters = kwargs.get('overriddenParameters', [])
        if len(self.overriddenParameters) > 0:
            self.overriddenParameters.sort(key=lambda tup: tup[0])
        self.normaliseAudio = kwargs.get('normaliseAudio', False)

        self.renderedPatch = False
        self.parameters = None
        self.patch = None


    @abstractmethod
    def setPatch(self, parameters):
        """
        Must be overridden. Should update parameters to values indicated.
        Should not effect the overridden parameters.

        :param parameters: A list of tuples. Tuples within the list must have the form
            `(parameter_index, parameter_value)` where parameter_index is an int with
            the parameter to modify and the parameter value is a float between 0-1.
            Can be a partial list of parameters for the synthesizer.
        :type parameters: list
        """
        pass


    @abstractmethod
    def renderPatch(self):
        """
        Must be overridden. Should render audio for the currently loaded patch
        """
        pass


    @abstractmethod
    def getAudio(self):
        """
        This method must be overridden and should return an audio buffer rendered
        durinf the last call to renderPatch.

        :return: An audio buffer of float audio samples with a value between -1 & 1
        :rtype: np.array
        """
        pass

    @abstractmethod
    def randomizePatch(self):
        """
        This method must be overridden and should have the effect
        of randomizing parameters of the synthesizer. Overridden methods should be
        uneffected by this randomization
        """
        pass


    def getRandomExample(self):
        """
        Returns audio from a new random patch

        :return: An audio buffer
        :rtype: np.array
        """

        self.randomizePatch()
        self.renderPatch()
        return self.getAudio()


    def getParameters(self):
        """
        Returns parameters for the synth

        :return: A dictionary of parameters with the parameter index (int) as the key
            and the parameter name short description as the value
        :rtype: Dictionary
        """

        return self.parameters


    def getPatch(self, skipOverridden=True):
        """
        Get current patch

        :param skipOverridden: Indicates whether to remove overridden parameters from results,
            defaults to True
        :type skipOverridden: bool, optional
        """

        if not skipOverridden:
            return self.patch

        patch = []
        overriddenParams = [p[0] for p in self.overriddenParameters]
        for item in self.patch:
            if not (item[0] in overriddenParams):
                patch.append(item)

        return patch
