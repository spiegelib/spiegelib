#!/usr/bin/env python
"""
Data scaler class that implments standarization. This transforms data
to have a mean of zero and variance of 1.
"""
import numpy as np
from spiegelib.features import DataScalerBase

class StandardScaler(DataScalerBase):
    """
    """

    def _reset(self):
        """
        Reset attributes
        """

        if hasattr(self, 'mean'):
            del self.mean
            del self.std


    def fit(self, data, axis=None):
        """
        Compute mean and std for later scaling

        :param data: data to use to calculate mean and std on
        :type data: np.ndarray
        :param axis: axis or axes to use for calculating scaling parameters on.
        :type axis: int, tuple
        """

        self._reset()
        self.mean = data.mean(axis)

        # Calculate standard deviation and handle zeros
        variance = data.var(axis)
        if isinstance(variance, np.ndarray):
            variance[variance == 0.0] = 1.0
            self.std = np.sqrt(variance)

        elif np.isscalar(variance):
            variance = 1.0 if variance == 0.0 else variance
            self.std = np.sqrt(variance)

        self.fit_axis = axis
        if self.fit_axis != None and not isinstance(self.fit_axis, tuple):
            self.fit_axis = (self.fit_axis,)
        self.fit_shape = data.shape


    def transform(self, data):
        """
        Perform normalization on data

        :param data: data to normalize
        :type data: np.array
        :returns: Normalized data
        :rtype: np.ndarray
        """

        if not hasattr(self, 'mean'):
            raise Exception("You must fit this scaler first")

        if self.fit_axis == None:
            return (data - self.mean) / self.std

        mean_expanded = np.copy(self.mean)
        std_expanded = np.copy(self.std)

        if len(data.shape) == len(self.fit_shape):
            for axis in self.fit_axis:
                mean_expanded = np.expand_dims(mean_expanded, axis=axis)
                std_expanded = np.expand_dims(std_expanded, axis=axis)

        elif len(data.shape) == len(self.fit_shape) - 1:
            for axis in self.fit_axis[1:]:
                mean_expanded = np.expand_dims(mean_expanded, axis=(axis-1))
                std_expanded = np.expand_dims(std_expanded, axis=(axis-1))

        else:
            raise ValueError("Input data has incorrect shape. Expected %sD or %D, received %D"
                             % (len(self.fit_shape), len(self.fit_shape) -1, len(data.shape)))

        return (data - mean_expanded) / std_expanded
