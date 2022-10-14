#!/usr/bin/env python

"""
:class:`SynthPlugin` is a class for interacting with synthesizer/fx plugins.

This class relies on the dawdreamer package. See https://github.com/DBraun/DawDreamer
"""

from __future__ import print_function
import random
import numpy as np
import dawdreamer as daw

from spiegelib import AudioBuffer
from spiegelib.synth.synth_base import SynthBase


class SynthPlugin(SynthBase):
    """
    :param plugin_path: path to vst plugin binary, defaults to None
    :type plugin_path: str, optional
    :param keyword arguments: see :class:`spiegelib.synth.synth_base.SynthBase` for details
    """

    def __init__(self, plugin_path=None, **kwargs):
        super().__init__(**kwargs)
        self.synth = None

        if plugin_path:
            self.load_plugin(plugin_path)

        else:
            self.engine = None
            self.loaded_plugin = False

    def load_plugin(self, plugin_path):
        """
        Loads a synth VST plugin

        :param plugin_path: path to vst plugin binary
        :type plugin_path: str
        """

        self.engine = daw.RenderEngine(self.sample_rate, self.buffer_size)
        self.synth = self.engine.make_plugin_processor("synth", plugin_path)

        # Make sure the synth was loaded with the expected name
        assert self.synth.get_name() == "synth"

        # Load the synth plugin into the dawdreamer processing graph
        graph = [(self.synth, [])]
        self.engine.load_graph(graph)

        self.loaded_plugin = True
        self.patch = [
            (i, self.synth.get_parameter(i))
            for i in range(self.synth.get_plugin_parameter_size())
        ]
        self.parameters = parse_parameters(self.synth.get_parameters_description())

        # TODO: Overridden parameters
        # for i in range(len(self.overridden_params)):
        #     index, value = self.overridden_params[i]
        #     self.engine.override_plugin_parameter(int(index), value)

    def load_patch(self):
        """
        Update patch parameter. Overridden parameters will not be effected.
        """

        # Check for parameters to include in patch update and set value
        for param in self.patch:
            if not self.is_valid_parameter_setting(param):
                raise Exception(
                    'Parameter %s is invalid. Must be a valid '
                    'parameter number and be in range 0-1. '
                    'Received %s' % param
                )
            self.synth.set_parameter(param[0], param[1])

    def is_valid_parameter_setting(self, parameter):
        """
        Checks to see if a parameter is valid for the currently loaded synth.

        :param parameter: A parameter tuple with form `(parameter_index, parameter_value)`
        :type parameter: tuple
        """
        return (
                parameter[0] in self.parameters
                and 0.0 <= parameter[1] <= 1.0
        )

    def render_patch(self):
        """
        Render the current patch. Uses the values of midi_note, midi_velocity, note_length_secs,
        and render_length_secs to render audio. Plugin must be loaded first.
        """
        if self.loaded_plugin:
            self.synth.clear_midi()
            self.synth.add_midi_note(self.midi_note, self.midi_velocity, 0.0, self.note_length_secs)
            self.engine.render(self.render_length_secs)
            self.rendered_patch = True

        else:
            print("Please load plugin first.")

    def get_audio(self):
        """
        Return monophonic audio from rendered patch

        :return: An audio buffer of the rendered patch
        :rtype: :class:`spiegelib.core.audio_buffer.AudioBuffer`
        """

        if self.rendered_patch:
            # TODO: Need to handle both stereo and mono audio
            audio = AudioBuffer(self.engine.get_audio()[0], self.sample_rate)
            return audio

        else:
            raise Exception('Patch must be rendered before audio can be retrieved')

    def randomize_patch(self):
        """
        Randomize the current patch. Overridden parameters will be unaffected.
        """
        # TODO: Need to handle the overridden patches!
        if self.loaded_plugin:
            random_patch = [(p[0], random.random()) for p in self.get_patch()]
            self.set_patch(random_patch)

        else:
            print("Please load plugin first.")



################################################################################

def parse_parameters(params):
    """
    Parse parameter dictionary from DawDreamer plugin processor into a dictionary with
    format {parameter_index: parameter_name}

    :param params: A dictionary returned by DawDreamer that contains detailed information about
        parameters in a plugin.
    :type params: dict
    :returns: A dictionary with parameter index as keys and parameter name / description for values
    :rtype: dict
    """

    param_dict = {}
    for param in params:
        param_dict[param['index']] = param['name']

    return param_dict
