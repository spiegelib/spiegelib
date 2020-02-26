#!/usr/bin/env python
"""
Abstract Base Class for evaluating audio files
"""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import json
from spiegel import AudioBuffer

class EvaluationBase(ABC):
    """
    Pass in a list of targets and lists of predictions for each target.

    Ex) targetList = [y1, y2]
        estimatedList = [[y1_estimation1, y1_estimation2], [y2_estimation1, y2_estimation2]]

    :param targetList: list of objects to use as the ground truth for evaluation.
    :type targetList: list
    :param estimatedList: List of lists of objects. There must be as many lists
        as there are targets, and each of those lists contain objects that are
        estimations for the associated target objects.
    :type estimatedList: list
    """

    def __init__(self, targetList, estimatedList):
        """
        Constructor
        """

        if not isinstance(targetList, list):
            raise TypeError("Expected targetList to be a list")

        self.verifyInputList(targetList)

        self.numTargets = len(targetList)
        self.targetList = targetList

        if self.numTargets != len(estimatedList):
            raise ValueError("Estimated list must contain same number of lists as targets")

        for item in estimatedList:
            if not isinstance(item, list):
                raise TypeError("Expected list of lists of AudioBuffers for estimatedList")

            self.verifyInputList(item)

        self.estimatedList = estimatedList
        self.scores = {}


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


    def getSummarizedScores(self):
        """
        :returns: A dictionary of scores summarized using mean, median, and stddev
            for each estimation
        :rtype: dict
        """
        pass



    def evaluate(self):
        """
        Run evaluation. Calls evaluateTarget on all targets and creates a dictionary
        of metrics stored in self.scores
        """

        self.scores = {}
        for i in range(len(self.targetList)):
             results = self.evaluateTarget(self.targetList[i], self.estimatedList[i])
             resultsDict = {}
             for j in range(len(results)):
                 resultsDict['estimation_%s' % j] = results[j]

             self.scores['target_%i' % i] = resultsDict


    @abstractmethod
    def evaluateTarget(self, target, predictions):
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
    def euclidianDistance(A, B):
        """
        Calculates the euclidian distance between two arrays.

        :param A: First array (Ground Truth)
        :type A: np.ndarray
        :param B: Second array (Prediction)
        :type B: np.ndarray
        :returns: absolue mean error value
        :rtype: float
        """
        return np.linalg.norm(np.subtract(A,B).flatten())


    @staticmethod
    def manhattanDistance(A, B):
        """
        Calculates the manhattan distance between two arrays.

        :param A: First array (Ground Truth)
        :type A: np.ndarray
        :param B: Second array (Prediction)
        :type B: np.ndarray
        :returns: absolue mean error value
        :rtype: float
        """
        return np.linalg.norm(np.subtract(A,B).flatten(), 1)



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


    def plotHist(self, estimations, metric, bins=None, **kwargs):
        """
        """

        values = []
        for estimation in estimations:
            values.extend([self.scores[key][estimation][metric] for key in self.scores])

        plt.hist(np.array(values), bins, facecolor='blue', alpha=0.75, edgecolor='black')
        plt.title(metric)
        plt.show()


    def saveScoresJSON(self, path):
        """
        """

        with open(path, 'w') as fp:
            json.dump(self.scores, fp, indent=True, cls=NumpyNumberEncoder)



    def verifyInputList(self, inputList):
        """
        Base method for verifying input list. Override to implement verification.
        """
        pass


    @staticmethod
    def verifyAudioInputList(inputList):
        """
        Static method for verifying input lists with audio buffers
        """

        for item in inputList:
            if not isinstance(item, AudioBuffer):
                raise TypeError('Must be an AudioBuffer, received %s' % type(item))



class NumpyNumberEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj).__module__ == np.__name__:
            try:
                return float(obj)
            except TypeError:
                pass
        return json.JSONEncoder.default(self, obj)
