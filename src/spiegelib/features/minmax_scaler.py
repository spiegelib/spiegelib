#!/usr/bin/env python
"""
Data scaler class that implments minmax normalization.

Scaler must be 'trained' (fit) before it can be used to scale new results.
"""
import numpy as np
from spiegelib.features import DataScalerBase

class MinMaxScaler(DataScalerBase):

    def _reset(self):
        """
        Reset attributes
        """

        if hasattr(self, 'min'):
            del self.min
            del self.max


    def fit(self, data, axis=None):
        """
        Compute mean and std for later scaling

        Args:
            data (np.ndarray): data to calculate mean and standard deviation on
                for future scaling
            axis (int, tuple, optional): axis or axes to use for calculating scaling
                parameteres. Defaults to None which will flatten the array first.
        """

        self._reset()
        self.min = data.min(axis)
        self.max = data.max(axis)

        self.fit_axis = axis
        if self.fit_axis != None and not isinstance(self.fit_axis, tuple):
            self.fit_axis = (self.fit_axis,)
        self.fit_shape = data.shape


    def transform(self, data):
        """
        Scale data

        Args:
            data (np.ndarray): data to scale

        Returns:
            np.ndarray: scaled data
        """

        if not hasattr(self, 'min'):
            raise Exception("You must fit this scaler first")

        if self.fit_axis is None:
            return (data - self.min) / (self.max - self.min)

        min_expanded = np.copy(self.min)
        max_expanded = np.copy(self.max)

        if len(data.shape) == len(self.fit_shape):
            for axis in self.fit_axis:
                min_expanded = np.expand_dims(min_expanded, axis=axis)
                max_expanded = np.expand_dims(max_expanded, axis=axis)

        elif len(data.shape) == len(self.fit_shape) - 1:
            for axis in self.fit_axis[1:]:
                min_expanded = np.expand_dims(min_expanded, axis=(axis-1))
                max_expanded = np.expand_dims(max_expanded, axis=(axis-1))

        else:
            raise ValueError("Input data has incorrect shape. Expected %s or %s, received %s"
                             % (len(self.fit_shape), len(self.fit_shape) -1, len(data.shape)))

        return (data - min_expanded) / (max_expanded - min_expanded)
