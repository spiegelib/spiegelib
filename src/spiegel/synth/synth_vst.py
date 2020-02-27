#!/usr/bin/env python

"""
:class:`SynthVST` is a class for interacting with VST synthesizer plugins.

This class relies on the librenderman package developed by Leon Feddon for interacting
with VSTs. A VST can be loaded, parameters displayed and modified, random patches generated,
and audio rendered for further processing.
"""

from __future__ import print_function
import numpy as np
from spiegel import AudioBuffer
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
            self.engine = rm.RenderEngine(self.sample_rate, self.bufferSize, self.bufferSize)

            if self.engine.load_plugin(pluginPath):
                self.loadedPlugin = True
                self.generator = rm.PatchGenerator(self.engine)
                self.patch = self.engine.get_patch()
                self.parameters = parseParameters(self.engine.get_plugin_parameters_description())

                for i in range(len(self.overriddenParameters)):
                    index, value = self.overriddenParameters[i]
                    self.engine.override_plugin_parameter(int(index), value)

            else:
                raise Exception('Could not load VST at path: %s' % pluginPath)

        except Exception as error:
            print(error)


    def loadPatch(self):
        """
        Update patch parameter. Overridden parameters will not be effected.
        """

        # Check for parameters to include in patch update
        for param in self.patch:
            if not self.isValidParameterSetting(param):
                raise Exception(
                    'Parameter %s is invalid. Must be a valid '
                    'parameter number and be in range 0-1. '
                    'Received %s' % param
                )

        # Patch VST with parameters
        self.engine.set_patch(self.patch)


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



    def render_patch(self):
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


    def get_audio(self):
        """
        Return monophonic audio from rendered patch

        :return: An audio buffer of the rendered patch
        :rtype: :class:`spiegel.core.audio_buffer.AudioBuffer`
        """

        if self.renderedPatch:
            audio = AudioBuffer(self.engine.get_audio_frames(), self.sample_rate)
            return audio

        else:
            raise Exception('Patch must be rendered before audio can be retrieved')


    def randomizePatch(self):
        """
        Randomize the current patch. Overridden parameteres will be unaffected.
        """

        if self.loadedPlugin:
            randomPatchTuples = self.generator.get_random_patch()
            self.set_patch(randomPatchTuples)

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
