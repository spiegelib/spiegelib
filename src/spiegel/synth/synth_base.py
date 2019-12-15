#!/usr/bin/env python
"""
Synth Abstract Base Class
"""
from __future__ import print_function
from abc import ABC, abstractmethod


class SynthBase(ABC):
    """ Base class for synths """

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
        self.warningMode = kwargs.get('warningMode', "always")
        self.normaliseAudio = kwargs.get('normaliseAudio', False)
        self.renderedPatch = False
        self.parameters = None
        self.patch = None


    def getParameters(self):
        """ Returns a list of parameters for this synth """
        return self.parameters


    @abstractmethod
    def setPatch(self, parameters):
        """ Update patch parameters """
        pass


    @abstractmethod
    def renderPatch(self):
        """ Call to render the current patch """
        pass


    @abstractmethod
    def getAudio(self):
        """ Return audio from rendered patch """
        pass
