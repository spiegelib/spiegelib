#!/usr/bin/env python
"""
Abstract Base Class for evaluating audio files
"""

from abc import ABC, abstractmethod
from spiegel import AudioBuffer

class AudioEvalBase(ABS):
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
    """

    def __init__(self, targetList, estimatedList):
        """
        Constructor
        """

        if not AudioEvalBase.verifyInputList(targetList):
            raise TypeError("All target list items must be of type AudioBuffer")

        self.numTargets = len(targetList)
        self.targetList = targetList

        if self.numTargets != len(estimatedList):
            raise ValueError("Estimated list must contain same number of lists as targets")

        for item in estimatedList:
            if not isinstance(item, list):
                raise TypeError("Expected list of lists of AudioBuffers for estimatedList")

            if not AudioEvalBase.verifyInputList(item):
                raise TypeError("All estimated list items must be of type AudioBuffer")

        self.estimatedList = estimatedList
        self.scores = {}


    def getScores():
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
    def evaluate():
        """
        Abstract method. Must be implemented and run evaluation. Results should be
        stored in scores member variable.
        """
        pass


    @staticmethod
    def verifyInputList(inputList):
        """
        Base method for verifying input list
        """

        for item in inputList:
            if not isinstance(item, AudioBuffer):
                return False

        return True
