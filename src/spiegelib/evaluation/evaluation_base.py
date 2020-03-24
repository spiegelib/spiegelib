#!/usr/bin/env python
"""
Abstract Base Class for evaluating audio files
"""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import json
from spiegelib import AudioBuffer

class EvaluationBase(ABC):
    """
    Pass in a list of targets and lists of predictions for each target.

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

        for item in estimations:
            if not isinstance(item, list):
                raise TypeError("Expected list of lists of AudioBuffers for "
                                "estimations")

            self.verify_input_list(item)

        self.num_sources = len(estimations)
        self.estimations = estimations
        self.scores = {}
        self.stats = {}


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


    def get_stats(self):
        """
        Return dictionary of stats that summarize the scores for each source
        """

        if not len(self.stats):
            self.compute_stats()

        return self.stats


    def compute_stats(self):
        """
        :returns: A dictionary of scores summarized using mean, median, and stddev
            for each source
        :rtype: dict
        """

        summaries = self.scores_to_nparray()

        # Now compute stats on those lists
        self.stats = {}
        for source in summaries:
            if not source in self.stats:
                self.stats[source] = {}

            for metric in summaries[source]:
                if not metric in self.stats[source]:
                    self.stats[source][metric] = {}

                self.stats[source][metric]['mean'] = np.median(summaries[source][metric])
                self.stats[source][metric]['median'] = np.mean(summaries[source][metric])
                self.stats[source][metric]['std'] = np.std(summaries[source][metric])
                self.stats[source][metric]['min'] = np.min(summaries[source][metric])
                self.stats[source][metric]['max'] = np.max(summaries[source][metric])


    def evaluate(self):
        """
        Run evaluation. Calls evaluate_target on all targets and creates a dictionary
        of metrics stored in self.scores
        """

        self.stats = {}
        self.scores = {}
        for i in range(len(self.targets)):
            target_estmations = [est[i] for est in self.estimations]
            results = self.evaluate_target(self.targets[i], target_estmations)
            results_dict = {}
            for j in range(len(results)):
                results_dict['source_%s' % j] = results[j]

            self.scores['target_%i' % i] = results_dict


    @abstractmethod
    def evaluate_target(self, target, predictions):
        """
        Abstract method. Must be implemented and run evaluation. Results should be
        stored in scores member variable.
        """
        pass


    @staticmethod
    def mean_abs_error(A, B):
        """
        Calculates mean absolute error between two arrays. Mean(ABS(A-B)).

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

    def scores_to_nparray(self):
        """
        Reorganize scores into a set of np.ndarrays, one for each metric for each
        source. Returned as a dictionary: dict[source][metric]
        """

        summaries = {}
        scores = self.get_scores()
        i = 0

        # Organize scores into arrays organized by the source and the metric
        for target in scores:
            for source in scores[target]:
                if not source in summaries:
                    summaries[source] = {}

                for metric in scores[target][source]:
                    if not metric in summaries[source]:
                        summaries[source][metric] = np.zeros(self.num_targets)

                    summaries[source][metric][i] = scores[target][source][metric]

            i = i + 1

        return summaries


    def plot_hist(self, sources, metric, bins=None, clip_upper=None, **kwargs):
        """
        """
        values = []
        for source in sources:
            values.extend([self.scores[key]['source_%s' % source][metric]
                           for key in self.scores])

        values = np.array(values)

        if clip_upper:
            values = np.clip(values, np.min(values), clip_upper)

        plt.hist(np.array(values), bins,
                 alpha=0.9, edgecolor='black')


    def save_scores_json(self, path):
        """
        """

        with open(path, 'w') as fp:
            json.dump(self.scores, fp, indent=True, cls=NumpyNumberEncoder)


    def save_stats_json(self, path):
        """
        Save score statistics as a JSON file

        :param path: Location of file to save json
        :type path: str
        """

        # Try to compute stats if it looks like they haven't been
        if not len(self.stats):
            self.compute_stats()

        # If there are now stats, save those
        if len(self.stats):
            with open(path, 'w') as fp:
                json.dump(self.stats, fp, indent=True, cls=NumpyNumberEncoder)

        else:
            print('No stats to save! Did you run evaluation?')


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
