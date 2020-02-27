#!/usr/bin/env python
"""
Synth Abstract Base Class
"""
from __future__ import print_function
import os
import numbers
import json
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
        self.midiNote = kwargs.get('midiNote', 48)
        self.midiVelocity = kwargs.get('midiVelocity', 127)
        self.noteLengthSecs = kwargs.get('noteLengthSecs', 1.0)
        self.renderLengthSecs = kwargs.get('renderLengthSecs', 2.5)
        self.overriddenParameters = kwargs.get('overriddenParameters', [])
        if len(self.overriddenParameters) > 0:
            self.overriddenParameters.sort(key=lambda tup: tup[0])
        self.normaliseAudio = kwargs.get('normaliseAudio', False)
        self.clampParams = kwargs.get('clampParams', True)

        self.renderedPatch = False
        self.parameters = None
        self.patch = None

        # Parameter range defaults to [0.0, 1.0]
        self.paramRange = (0.0, 1.0)


    def set_patch(self, parameters):
        """
        Set a new patch with updated parameter values with a list of flaots or a
        list of tuples referring to parameter indices and values.

        :param parameters: A list of floats or tuples. If passing in a list of numbers,
            the length must be either the number of non-overridden parameters of the
            number of total parameters for this synth. Parameter values will then be
            mapped to corresponding parameter in order. If a list of tuples is provided,
            the tuples must have the shape (int, float) where the int is the parameter
            number and the float is the parameter value.
        :type parameters: list
        """

        if not len(parameters):
            return

        parameterIndices = [p for p in self.getParameters()]
        overriddenIndices = [p[0] for p in self.overriddenParameters]
        nonOverriddenIndices = list(set(parameterIndices) - set(overriddenIndices))
        newPatch = []

        # If this is just a list of numbers, then try to associate with parameter settings
        if isinstance(parameters[0], numbers.Number):

            # Received same number of parameters as non-overridden parameters,
            # map directly to non-overridden parameters
            if len(parameters) == len(nonOverriddenIndices):
                newPatch = [
                    (nonOverriddenIndices[i], float(parameters[i])) for i in range(len(parameters))
                ]

            # Received same number of parameters as total parameter count,
            # map the non-overridden parameters from that list
            elif len(parameters) == len(parameterIndices):
                    newPatch = [
                        (i, float(parameters[i])) for i in nonOverriddenIndices
                    ]

            else:
                raise Exception((
                    'Unclear on how to map parameters, received %s parameters '
                    'and there are %s non-overridden parameters and %s total parameters.'
                ) % (len(parameters), len(nonOverriddenIndices), len(overriddenIndices)))

        # If this is a list of tuples then add those directly
        elif len(parameters[0]) == 2:
            newPatch = parameters

        else:
            raise Exception('Invalid parameter list provided. Must be a list of numbers or a list of tuples.')

        # Update patch member variable with new patch, skipping any overridden params
        for param in newPatch:
            if not param[0] in overriddenIndices:
                value = param[1]
                if self.clampParams:
                    value = max(min(value, self.paramRange[1]), self.paramRange[0])

                self.patch[param[0]] = (param[0], value)

        # Load new patch into synth engine
        self.renderedPatch = False
        self.loadPatch()


    @abstractmethod
    def loadPatch(self):
        """
        Must be overridden. Load current patch into synth engine
        """
        pass


    @abstractmethod
    def render_patch(self):
        """
        Must be overridden. Should render audio for the currently loaded patch
        """
        pass


    @abstractmethod
    def get_audio(self):
        """
        This method must be overridden and should return an audio buffer rendered
        during the last call to render_patch.

        :return: An audio buffer of float audio samples with a value between -1 & 1
        :rtype: :class:`spiegel.core.audio_buffer.AudioBuffer`
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
        self.render_patch()
        return self.get_audio()


    def getParameters(self):
        """
        Returns parameters for the synth

        :return: A dictionary of parameters with the parameter index (int) as the key
            and the parameter name short description as the value
        :rtype: Dictionary
        """

        return self.parameters


    def get_patch(self, skipOverridden=True):
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


    def setOverriddenParameters(self, parameters):
        """
        Override parameters with specific values

        :param parameters: List of parameter tuples with parameter index and parameter
            value that will be patched and then the parameter frozen.
        :type parameter: list
        """

        overriddenParams = []
        for param in parameters:
            overriddenParams.append((param[0], param[1]))

        self.overriddenParameters = []
        self.set_patch(overriddenParams)
        self.overriddenParameters = overriddenParams



    def saveParameterState(self, path):
        """
        Save parameters and current state to a JSON file. Includes whether or not
        a parameter has been overridden so this can be used to save synth configuration

        :param path: Location to save JSON file to
        :type path: str
        """

        assert self.parameters, "Parameters must be set before saving parameter state"
        assert self.patch,      "Patch must be set before saving parameter state"

        # Make sure directory exists and create if it doesn't
        fullPath = os.path.abspath(path)
        directory = os.path.dirname(fullPath)

        if not os.path.exists(directory):
            os.mkdir(directory)

        # Create a dictionary of parameters and settings for saving
        parameterDict = {}
        overriddenParams = [p[0] for p in self.overriddenParameters]

        for parameter in self.patch:
            parameterDict[parameter[0]] = {
                "id": parameter[0],
                "desc": self.parameters[parameter[0]],
                "value": float(parameter[1]),
                "overridden": parameter[0] in overriddenParams
            }

        with open(fullPath, 'w') as fp:
            json.dump(parameterDict, fp, indent=True, sort_keys=True)


    def loadParameterState(self, path):
        """
        Load parameter state from JSON file. Will set a new patch and overridden
        parameters.

        :param path: Location to load JSON file from
        :type path: str
        """

        fullPath = os.path.abspath(path)
        assert os.path.exists(fullPath), "Path does not exists: %s" % fullPath

        parameterDict = {}
        with open(fullPath, 'r') as fp:
            parameterDict = json.load(fp)

        patch = []
        overridden = []
        for key in parameterDict:
            if parameterDict[key]['overridden']:
                overridden.append((
                    parameterDict[key]['id'],
                    parameterDict[key]['value']
                ))
            else:
                patch.append((
                    parameterDict[key]['id'],
                    parameterDict[key]['value']
                ))

        self.setOverriddenParameters(overridden)
        self.set_patch(patch)
