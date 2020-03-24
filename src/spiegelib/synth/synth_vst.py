#!/usr/bin/env python

"""
:class:`SynthVST` is a class for interacting with VST synthesizer plugins.

This class relies on the librenderman package developed by Leon Feddon for interacting
with VSTs. A VST can be loaded, parameters displayed and modified, random patches generated,
and audio rendered for further processing.
"""

from __future__ import print_function
import numpy as np
import librenderman as rm

from spiegelib import AudioBuffer
from spiegelib.synth.synth_base import SynthBase


class SynthVST(SynthBase):
    """
    :param plugin_path: path to vst plugin binary, defaults to None
    :type plugin_path: str, optional
    :param keyword arguments: see :class:`spiegelib.synth.synth_base.SynthBase` for details
    """

    def __init__(self, plugin_path=None, **kwargs):
        super().__init__(**kwargs)

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
        try:
            self.engine = rm.RenderEngine(self.sample_rate, self.buffer_size, self.buffer_size)

            if self.engine.load_plugin(plugin_path):
                self.loaded_plugin = True
                self.generator = rm.PatchGenerator(self.engine)
                self.patch = self.engine.get_patch()
                self.parameters = parse_parameters(self.engine.get_plugin_parameters_description())

                for i in range(len(self.overridden_params)):
                    index, value = self.overridden_params[i]
                    self.engine.override_plugin_parameter(int(index), value)

            else:
                raise Exception('Could not load VST at path: %s' % plugin_path)

        except Exception as error:
            print(error)


    def load_patch(self):
        """
        Update patch parameter. Overridden parameters will not be effected.
        """

        # Check for parameters to include in patch update
        for param in self.patch:
            if not self.is_valid_parameter_setting(param):
                raise Exception(
                    'Parameter %s is invalid. Must be a valid '
                    'parameter number and be in range 0-1. '
                    'Received %s' % param
                )

        # Patch VST with parameters
        self.engine.set_patch(self.patch)


    def is_valid_parameter_setting(self, parameter):
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
        Render the current patch. Uses the values of midi_note, midi_velocity, note_length_secs,
        and render_length_secs to render audio. Plugin must be loaded first.
        """
        if self.loaded_plugin:
            self.engine.render_patch(
                self.midi_note,
                self.midi_velocity,
                self.note_length_secs,
                self.render_length_secs
            )
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
            audio = AudioBuffer(self.engine.get_audio_frames(), self.sample_rate)
            return audio

        else:
            raise Exception('Patch must be rendered before audio can be retrieved')


    def randomize_patch(self):
        """
        Randomize the current patch. Overridden parameteres will be unaffected.
        """

        if self.loaded_plugin:
            random_patch = self.generator.get_random_patch()
            self.set_patch(random_patch)

        else:
            print("Please load plugin first.")



################################################################################


def parse_parameters(param_str):
    """
    Parse parameter string return by librenderman into a dictionary keyed on parameter
    index with values being the name / short descriptions for the parameter at that index.

    :param param_str: A parameter decription string returned by librenderman
    :type param_str: str
    :returns: A dictionary with parameter index as keys and parameter name / description for values
    :rtype: dict
    """

    param_list = param_str.split("\n")
    param_dict = {}

    for param in param_list:
        param_vals = param.split(":")
        try:
            param_index = int(param_vals[0])
            param_name = param_vals[1].strip()
            param_dict[param_index] = param_name
        except ValueError:
            continue

    return param_dict
