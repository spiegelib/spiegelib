#!/usr/bin/env python
"""
Synth Abstract Base Class
"""

import os
import numbers
import json
from abc import ABC, abstractmethod


class SynthBase(ABC):
    """
    :param sample_rate: sampling rate for rendering audio, defaults to 44100
    :type sample_rate: int, optional
    :param buffer_size: buffer size for rendering audio, defaults to 512
    :type buffer_size: int, optional
    :param midi_note: midi note number used for rendering, 0-127, defaults to 40
    :type midi_note: int, optional
    :param midi_velocity: midi velocity used for rendering. 0-127, defaults to 127
    :type midi_velocity: int, optional
    :param note_length_secs: length of midi note in seconds, defaults to 1.0
    :type note_length_secs: float, optional
    :param render_length_secs: length that audio is rendered for in total, defaults to 2.5
    :type render_length_secs: float, optional
    :param overridden_params: a list of tuples containing the parameter index to override and the value to lock that parameter to, defaults to []
    :type overridden_params: list, optional
    :param clamp_params: If true, parameter values will be forced to the acceptable
        range of values. Defaults to True.
    :type clamp_params: bool, optional

    :cvar rendered_patch: indicates whether a patch has been rendered yet, defaults to False
    :vartype rendered_patch: boolean
    :cvar parameters: parameter indices and names
    :vartype parameters: dict
    :cvar patch: current patch values
    :vartype patch: list
    :cvar param_range: Range of acceptable parameter values, defaults to (0.0, 1.0)
    :vartype param_range: tuple
    """

    def __init__(self, sample_rate=44100, buffer_size=512, midi_note=48,
                 midi_velocity=127, note_length_secs=1.0, render_length_secs=2.0,
                 overridden_params=None, clamp_params=True):
        """
        Constructor
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.midi_note = midi_note
        self.midi_velocity = midi_velocity
        self.note_length_secs = note_length_secs
        self.render_length_secs = render_length_secs
        self.overridden_params = overridden_params or []
        self.clamp_params = clamp_params

        # Whether or not patch has been rendered by engine
        self.rendered_patch = False

        # Parameters for this synthesizer, must be set
        self.parameters = None

        # Current patch
        self.patch = None

        # Parameter range defaults to [0.0, 1.0]
        self.param_range = (0.0, 1.0)


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

        if len(parameters) == 0:
            return

        param_indices = self.get_parameters().keys()
        overridden_indices = [p[0] for p in self.overridden_params]
        new_patch = []

        # If this is just a list of numbers, then try to associate with parameter settings
        if isinstance(parameters[0], numbers.Number):
            new_patch = SynthBase.expand_sub_patch(parameters,
                                                   param_indices,
                                                   self.overridden_params)

        # If this is a list of tuples then add those directly
        elif len(parameters[0]) == 2:
            new_patch = parameters

        else:
            raise Exception("Invalid parameter list provided. Must be a list of "
                            "numbers or a list of tuples.")

        # Update patch member variable with new patch, skipping any overridden params
        for param in new_patch:
            if not param[0] in overridden_indices:
                value = param[1]
                if self.clamp_params:
                    value = max(min(value, self.param_range[1]), self.param_range[0])

                self.patch[param[0]] = (param[0], value)

        # Load new patch into synth engine
        self.rendered_patch = False
        self.load_patch()


    @abstractmethod
    def load_patch(self):
        """
        Must be overridden. Load current patch into synth engine
        """


    @abstractmethod
    def render_patch(self):
        """
        Must be overridden. Should render audio for the currently loaded patch
        """


    @abstractmethod
    def get_audio(self):
        """
        This method must be overridden and should return an audio buffer rendered
        during the last call to render_patch.

        :return: An audio buffer of float audio samples with a value between -1 & 1
        :rtype: :class:`spiegelib.core.audio_buffer.AudioBuffer`
        """


    @abstractmethod
    def randomize_patch(self):
        """
        This method must be overridden and should have the effect
        of randomizing parameters of the synthesizer. Overridden methods should be
        uneffected by this randomization
        """


    def get_random_example(self):
        """
        Returns audio from a new random patch

        :return: An audio buffer
        :rtype: np.array
        """

        self.randomize_patch()
        self.render_patch()
        return self.get_audio()


    def get_parameters(self):
        """
        Returns parameters for the synth

        :return: A dictionary of parameters with the parameter index (int) as the key
            and the parameter name short description as the value
        :rtype: Dictionary
        """

        return self.parameters


    def get_patch(self, skip_overridden=True):
        """
        Get current patch

        :param skip_overridden: Indicates whether to remove overridden parameters
            from results, defaults to True
        :type skip_overridden: bool, optional
        """

        if not skip_overridden:
            return self.patch

        patch = []
        overridden_params = [p[0] for p in self.overridden_params]
        for item in self.patch:
            if not item[0] in overridden_params:
                patch.append(item)

        return patch


    def set_overridden_parameters(self, parameters):
        """
        Override parameters with specific values

        :param parameters: List of parameter tuples with parameter index and parameter
            value that will be patched and then the parameter frozen.
        :type parameter: list
        """

        overridden_params = []
        for param in parameters:
            overridden_params.append((param[0], param[1]))

        self.overridden_params = []
        self.set_patch(overridden_params)
        self.overridden_params = overridden_params



    def save_state(self, path):
        """
        Save parameters and current state to a JSON file. Includes whether or not
        a parameter has been overridden so this can be used to save synth configuration

        :param path: Location to save JSON file to
        :type path: str
        """

        assert self.parameters, "Parameters must be set before saving parameter state"
        assert self.patch, "Patch must be set before saving parameter state"

        # Make sure directory exists and create if it doesn't
        fullpath = os.path.abspath(path)
        directory = os.path.dirname(fullpath)

        if not os.path.exists(directory):
            os.mkdir(directory)

        # Create a dictionary of parameters and settings for saving
        param_dict = {}
        overridden_params = [p[0] for p in self.overridden_params]

        for parameter in self.patch:
            param_dict[parameter[0]] = {
                "id": parameter[0],
                "desc": self.parameters[parameter[0]],
                "value": float(parameter[1]),
                "overridden": parameter[0] in overridden_params
            }

        with open(fullpath, 'w') as file_handle:
            json.dump(param_dict, file_handle, indent=True, sort_keys=True)


    def load_state(self, path):
        """
        Load parameter state from JSON file. Will set a new patch and overridden
        parameters.

        :param path: Location to load JSON file from
        :type path: str
        """

        patch, overridden = SynthBase.load_synth_config(path)
        self.set_overridden_parameters(overridden)
        self.set_patch(patch)


    @staticmethod
    def load_synth_config(path):
        """
        Loads and extracts patch setting and overridden parameters from a saved
        synth JSON file

        Args:
            path (str): location to load JSON file from

        Returns:
            list, list: patch list and overridden parameter list
        """

        fullpath = os.path.abspath(path)
        assert os.path.exists(fullpath), "Path does not exists: %s" % fullpath

        param_dict = {}
        with open(fullpath, 'r') as file_handle:
            param_dict = json.load(file_handle)

        patch = []
        overridden = []
        for key in param_dict:
            if param_dict[key]['overridden']:
                overridden.append((
                    param_dict[key]['id'],
                    param_dict[key]['value']
                ))
            else:
                patch.append((
                    param_dict[key]['id'],
                    param_dict[key]['value']
                ))

        return patch, overridden


    @staticmethod
    def expand_sub_patch(patch, param_indices, overridden):
        """
        Convert an ordered list of parameter values into a list of tuples containing
        the parameter indices linked to the parameter values. Does so by comparing
        the overridden parameters to the full list of paramater indices and using
        the differences to label the patch values.

        Args:
            patch (list): An ordered list of parameter values
            param_indices (list): A full list of all parameter indices
            overridden (list of tuples): A list of tuples which contain the parameter
                indices and values of the overridden parameters

        Returns:
            list: a list of tuples representing a full patch setting
        """

        overridden_indices = [p[0] for p in overridden]
        non_overridden_indices = list(set(param_indices) - set(overridden_indices))
        new_patch = []

        # Received same number of parameters as non-overridden parameters,
        # map directly to non-overridden parameters
        if len(patch) == len(non_overridden_indices):
            new_patch = [(non_overridden_indices[i], float(patch[i]))
                         for i in range(len(patch))]

        # Received same number of parameters as total parameter count,
        # map the non-overridden parameters from that list
        elif len(patch) == len(param_indices):
            new_patch = [(i, float(patch[i])) for i in non_overridden_indices]

        else:
            raise ValueError((
                'Unclear on how to map parameters, received %s parameters '
                'and there are %s non-overridden parameters and %s total parameters.'
            ) % (len(patch), len(non_overridden_indices),
                 len(overridden_indices + non_overridden_indices)))

        return new_patch
