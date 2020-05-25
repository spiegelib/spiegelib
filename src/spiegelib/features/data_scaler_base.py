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

        Args:
            data (np.ndarray): data to calculate mean and standard deviation on
                for future scaling
            axis (int, tuple, optional): axis or axes to use for calculating scaling
                parameteres. Defaults to None which will flatten the array first.
        """
        raise NotImplementedError


    @abstractmethod
    def transform(self, data):
        """
        Scale data

        Args:
            data (np.ndarray): data to scale

        Returns:
            np.ndarray: scaled data
        """
        raise NotImplementedError


    def fit_transform(self, data, axis=None):
        """
        Fit scaling parameters and then scale data

        Args:
            data (np.ndarray): data to scale
            axis (int, tuple, optional): axis or axes to use for calculating scaling
                parameteres. Defaults to None which will flatten the array first.

        Returns:
            np.ndarray: scaled data
        """

        self.fit(data, axis)
        return self.transform(data)
