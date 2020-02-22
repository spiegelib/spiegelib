#!/usr/bin/env python
"""
Abstract Base Class for evaluating audio files
"""

from abc import ABC, abstractmethod
import numpy as np
from spiegel import AudioBuffer

class AudioEvalBase(ABC):
    """
    Pass in a list of target AudioBuffers and lists of estimated AudioBuffers for
    each target AudioBuffer.

    Ex) targetList = [y1, y2]
        estimatedList = [[y1_estimation1, y1_estimation2], [y2_estimation1, y2_estimation2]]

    :param targetList: list of :class:`spiegel.core.audio_buffer.AudioBuffer` objects
        to use as the ground truth for evaluation.
    :type targetList: list
    :param estimatedList: List of lists of :class:`spiegel.core.audio_buffer.AudioBuffer` objects.
        There must be as many lists as there are targets, and each of those lists contain
        AudioBuffers that are estimations for the associated target AudioBuffer.
    :type estimatedList: list
    :param sampleRate: If set, will resample all input audio to that rate, otherwise will
        try to determine sample rate based on input AudioBuffers. Defaults to None
    :type sampleRate: optional, int, None
    """

    def __init__(self, targetList, estimatedList, sampleRate=None):
        """
        Constructor
        """

        if not isinstance(targetList, list):
            raise TypeError("Expected targetList to be a list")

        AudioEvalBase.verifyInputList(targetList)

        self.numTargets = len(targetList)
        self.targetList = targetList

        if self.numTargets != len(estimatedList):
            raise ValueError("Estimated list must contain same number of lists as targets")

        for item in estimatedList:
            if not isinstance(item, list):
                raise TypeError("Expected list of lists of AudioBuffers for estimatedList")

            AudioEvalBase.verifyInputList(item)

        self.estimatedList = estimatedList
        self.scores = {}
        self.sampleRate = sampleRate if sampleRate else self.targetList[0].getSampleRate()


    def getScores(self):
        """
        :returns: Return scores calculated during evaluation
        :rtype: dict
        """
        if not len(self.scores):
            print(
                "No scores available, did you run evaluate method? If you did "
                "make sure scores are updated in evaluate method."
            )

        return self.scores



    @abstractmethod
    def evaluate(self):
        """
        Abstract method. Must be implemented and run evaluation. Results should be
        stored in scores member variable.
        """
        pass


    @staticmethod
    def absoluteMeanError(A, B):
        """
        Calculates absolute mean error between two arrays. Mean(ABS(A-B)).

        :param A: First array (Ground Truth)
        :type A: np.ndarray
        :param B: Second array (Prediction)
        :type B: np.ndarray
        :returns: absolue mean error value
        :rtype: float
        """
        return np.mean(np.abs(A-B).flatten())


    @staticmethod
    def meanSquaredError(A, B):
        """
        Calculates mean squared error between two arrays. Mean(Square(A-B)).

        :param A: First array (Ground Truth)
        :type A: np.ndarray
        :param B: Second array (Prediction)
        :type B: np.ndarray
        :returns: absolue mean error value
        :rtype: float
        """
        return np.mean(np.square(A-B).flatten())


    @staticmethod
    def verifyInputList(inputList):
        """
        Base method for verifying input list
        """

        for item in inputList:
            if not isinstance(item, AudioBuffer):
                raise TypeError('Must be an AudioBuffer, received %s' % type(item))

        return True
