#!/usr/bin/env python
"""
Abstract Base Class for Data Scalers
"""

from abc import ABC, abstractmethod

class DataScalerBase(ABC):

    @abstractmethod
    def fit(self, data, axis=None):
        """
        Fit parameters for data scaling. Abstract, must be implemented.

        :param data: data to use for fitting scaling parameters
        :type data: np.ndarray
        :param axis: axis or axes to use for calculating scaling parameters on.
        :type axis: int, tuple
        """
        raise NotImplementedError


    @abstractmethod
    def transform(self, data):
        """
        Perform scaling on data

        :param data: data to scale
        :type data: np.array
        :returns: sacled data
        :rtype: np.ndarray
        """
        raise NotImplementedError


    def fit_transform(self, data, axis=None):
        """
        Fit scaling parameters and then scale data

        :param data: data to fit scaling parameters to and then transform
        :type data: np.array
        :returns: scaled data
        :rtype: np.ndarray
        """

        self.fit(data, axis)
        return self.transform(data)
