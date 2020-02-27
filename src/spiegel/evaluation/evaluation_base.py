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

    Ex) targets = [y1, y2]
        estimations = [[y1_estimation1, y1_estimation2], [y2_estimation1, y2_estimation2]]

    :param targets: list of objects to use as the ground truth for evaluation.
    :type targets: list
    :param estimations: List of lists of objects. There must be as many lists
        as there are targets, and each of those lists contain objects that are
        estimations for the associated target objects.
    :type estimations: list
    """

    def __init__(self, targets, estimations):
        """
        Constructor
        """

        if not isinstance(targets, list):
            raise TypeError("Expected targets to be a list")

        self.verify_input_list(targets)

        self.num_targets = len(targets)
        self.targets = targets

        if self.num_targets != len(estimations):
            raise ValueError("Estimated list must contain same number of "
                             "lists as targets")

        for item in estimations:
            if not isinstance(item, list):
                raise TypeError("Expected list of lists of AudioBuffers for "
                                "estimations")

            self.verify_input_list(item)

        self.estimations = estimations
        self.scores = {}


    def get_scores(self):
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


    def get_summarized_scores(self):
        """
        :returns: A dictionary of scores summarized using mean, median, and stddev
            for each estimation
        :rtype: dict
        """
        pass



    def evaluate(self):
        """
        Run evaluation. Calls evaluate_target on all targets and creates a dictionary
        of metrics stored in self.scores
        """

        self.scores = {}
        for i in range(len(self.targets)):
             results = self.evaluate_target(self.targets[i], self.estimations[i])
             results_dict = {}
             for j in range(len(results)):
                 results_dict['estimation_%s' % j] = results[j]

             self.scores['target_%i' % i] = results_dict


    @abstractmethod
    def evaluate_target(self, target, predictions):
        """
        Abstract method. Must be implemented and run evaluation. Results should be
        stored in scores member variable.
        """
        pass


    @staticmethod
    def abs_mean_error(A, B):
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
    def euclidian_distance(A, B):
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
    def manhattan_distance(A, B):
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
    def mean_squared_error(A, B):
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


    def plot_hist(self, estimations, metric, bins=None, **kwargs):
        """
        """

        values = []
        for estimation in estimations:
            values.extend([self.scores[key][estimation][metric]
                           for key in self.scores])

        plt.hist(np.array(values), bins, facecolor='blue',
                 alpha=0.75, edgecolor='black')
        plt.title(metric)
        plt.show()


    def save_scores_json(self, path):
        """
        """

        with open(path, 'w') as fp:
            json.dump(self.scores, fp, indent=True, cls=NumpyNumberEncoder)



    def verify_input_list(self, input_list):
        """
        Base method for verifying input list. Override to implement verification.
        """
        pass


    @staticmethod
    def verify_audio_input_list(input_list):
        """
        Static method for verifying input lists with audio buffers
        """

        for item in input_list:
            if not isinstance(item, AudioBuffer):
                raise TypeError('Must be an AudioBuffer, received %s' \
                                % type(item))



class NumpyNumberEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj).__module__ == np.__name__:
            try:
                return float(obj)
            except TypeError:
                pass
        return json.JSONEncoder.default(self, obj)
