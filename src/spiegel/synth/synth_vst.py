#!/usr/bin/env python

"""
:class:`SynthVST` is a class for interacting with VST synthesizer plugins.

This class relies on the librenderman package developed by Leon Feddon for interacting
with VSTs. A VST can be loaded, parameters displayed and modified, random patches generated,
and audio rendered for further processing.
"""

from __future__ import print_function
import numpy as np
from spiegel.synth.synth_base import SynthBase
import librenderman as rm


class SynthVST(SynthBase):
    """
    :param pluginPath: path to vst plugin binary, defaults to None
    :type pluginPath: str, optional
    :param keyword arguments: see :class:`spiegel.synth.synth_base.SynthBase` for details
    """

    def __init__(self, pluginPath=None, **kwargs):
        super().__init__(**kwargs)

        if pluginPath:
            self.loadPlugin(pluginPath)

        else:
            self.engine = None
            self.loadedPlugin = False


    def loadPlugin(self, pluginPath):
        """
        Loads a synth VST plugin

        :param pluginPath: path to vst plugin binary
        :type pluginPath: str
        """
        try:
            self.engine = rm.RenderEngine(self.sampleRate, self.bufferSize, self.bufferSize)

            if self.engine.load_plugin(pluginPath):
                self.loadedPlugin = True
                self.generator = rm.PatchGenerator(self.engine)
                self.parameters = parseParameters(self.engine.get_plugin_parameters_description())
                for i in range(len(self.overriddenParameters)):
                    index, value = self.overriddenParameters[i]
                    self.engine.override_plugin_parameter(int(index), value)

            else:
                raise Exception('Could not load VST at path: %s' % pluginPath)

        except Exception as error:
            print(error)


    def setPatch(self, parameters):
        """
        Update patch parameter. Overridden parameters will not be effected.

        :param parameters: A list of tuples. Tuples within the list must have the form
            `(parameter_index, parameter_value)` where parameter_index is an int with
            the parameter to modify and the parameter value is a float between 0-1.
            Can be a partial list of parameters for the synthesizer. See
            :func:`getParameters` to get parameter indices for the loaded synth.
        :type parameters: list
        """

        # Check for parameters to include in patch update
        parametersToPatch = []
        for param in parameters:
            if self.isValidParameterSetting(param):
                parametersToPatch.append(param)

        # Patch VST with parameters
        self.patch = parametersToPatch
        self.engine.set_patch(parametersToPatch)


    def isValidParameterSetting(self, parameter):
        """
        Checks to see if a parameter is valid for the currently loaded synth.

        :param parameter: A parameter tuple with form `(parameter_index, parameter_value)`
        :type parameter: tuple
        """
        return (
            parameter[0] in self.parameters
            and parameter[1] >= 0.0
            and parameter[1] <= 1.0
        )



    def renderPatch(self):
        """
        Render the current patch. Uses the values of midiNote, midiVelocity, noteLengthSecs,
        and renderLengthSecs to render audio. Plugin must be loaded first.
        """
        if self.loadedPlugin:
            self.engine.render_patch(
                self.midiNote,
                self.midiVelocity,
                self.noteLengthSecs,
                self.renderLengthSecs
            )
            self.renderedPatch = True

        else:
            print("Please load plugin first.")


    def getAudio(self):
        """
        Return monophonic audio from rendered patch

        :return: An audio signal of the rendered patch
        :rtype: 1D np.array
        """
        if self.renderedPatch:
            audio = np.array(self.engine.get_audio_frames())
            return audio

        else:
            print("Please render patch first.")
            return np.array([])


    def randomizePatch(self):
        """
        Randomize the current patch. Overridden parameteres will be unaffected.
        """

        if self.loadedPlugin:
            randomPatchTuples = self.generator.get_random_patch()
            self.setPatch(randomPatchTuples)

        else:
            print("Please load plugin first.")



################################################################################


def parseParameters(paramStr):
    """
    Parse parameter string return by librenderman into a dictionary keyed on parameter
    index with values being the name / short descriptions for the parameter at that index.

    :param paramStr: A parameter decription string returned by librenderman
    :type paramStr: str
    :returns: A dictionary with parameter index as keys and parameter name / description for values
    :rtype: dict
    """

    paramList = paramStr.split("\n")
    paramDict = {}

    for param in paramList:
        paramVals = param.split(":")
        try:
            paramIndex = int(paramVals[0])
            paramName = paramVals[1].strip()
            paramDict[paramIndex] = paramName
        except ValueError:
            continue

    return paramDict
