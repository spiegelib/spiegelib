#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import spiegel.synth.synthbase
import librenderman as rm


class SynthVST(spiegel.synth.synthbase.SynthBase):
    """ Class for interacting with a VST synthesizer """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.engine = None
        self.loadedPlugin = False


    def loadPlugin(self, pluginPath):
        """ Attempts loading a VST Synth """
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
        """ Update patch parameters """
        self.patch = {}


    def renderPatch(self):
        """ Call to render the current patch """
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
        """ Return audio from rendered patch """
        audio = np.array(self.engine.get_audio_frames())
        return audio


    def randomizePatch(self):
        """ Randomize the current patch """

        if self.loadedPlugin:
            randomPatchTuples = self.generator.get_random_patch()
            self.engine.set_patch(randomPatchTuples)

        else:
            print("Please load plugin first.")

        return



################################################################################


def parseParameters(paramStr):
    """ Parse parameters into dictionary """

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
