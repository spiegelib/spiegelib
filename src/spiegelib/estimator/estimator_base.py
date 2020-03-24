#!/usr/bin/env python
"""
Abstract Base Class for Synthesizer Parameter Estimators
"""

from abc import ABC, abstractmethod

class EstimatorBase(ABC):

    def __init__(self):
        """
        Constructor
        """
        pass


    @abstractmethod
    def predict(self):
        """
        Predict method must be implemented and should estimate parameters
        given some input
        """
        pass
