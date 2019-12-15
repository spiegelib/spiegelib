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
    :param normaliseAudio: whether or not to normalize rendered audio, default to False
    :type normaliseAudio: boolean, optional
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        self.patch = {}


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
            return np.array()


    def randomizePatch(self):
        """
        Randomize the current patch. Overridden parameteres will be unaffected.
        """

        if self.loadedPlugin:
            randomPatchTuples = self.generator.get_random_patch()
            self.engine.set_patch(randomPatchTuples)

        else:
            print("Please load plugin first.")

        return



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
