#!/usr/bin/env python
"""
Abstract Base Class for objective evaluations
"""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import json

from spiegelib import AudioBuffer
from spiegelib.core.utils import NumpyNumberEncoder

class EvaluationBase(ABC):
    """
    Example Input::

        targets = [target_1, target_2]
        estimations = [[prediction_1_for_target_1, prediction_2_for_target_1],
                       [prediction_1_for_target_2, prediction_2_for_target_2]]

    Where prediction_1 and prediction_2 would represent results from two different
    methods or sources. For example, audio for all prediction_1 samples might be
    from a GA and all audio for prediction_2 samples might be from a deep learning
    approach.

    Args:
        targets (list): a list of objects to use
            as the ground truth for evaluation
        estimations (lits): a 2D list of objects.
            Should contain a list of objects representing estimations for each target.
            The position of an object in each list is used to distinguish between different
            sources. For example, if you are comparing two different
            synthesizer programming methods, then you would want to make sure to
            have the results from each method in the same position in each list.
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
                raise TypeError("Expected list of lists for estimations")

            self.verify_input_list(item)

        self.num_sources = len(estimations)
        self.estimations = estimations
        self.scores = {}
        self.stats = {}


    @abstractmethod
    def evaluate_target(self, target, predictions):
        """
        Abstract method. Must be implemented and evaluate a single target and predictions
        made for that target. Called automatically by :func:`~evaluate`

        Args:
            target (list): Audio to use as ground truth in evaluation
            predictions (list): list of :ref:`AudioBuffer <audio_buffer>` objects to evaluate
                against the target audio file.
        Returns:
            list: A list of dictionaries with stats for each prediction evaluation
        """
        pass


    def evaluate(self):
        """
        Run evaluation. Calls evaluate_target on all targets and creates a dictionary
        of metrics stored in the scores attribute.

        Saves each prediction for a target in a dictionary keyed by the position in
        the prediction list that it was constructed with - uses the key 'source_#'.
        Where # is the position index.
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


    def get_scores(self):
        """
        Returns:
            dict: scores calculated during evaluation
        """
        if not len(self.scores):
            print(
                "No scores available, did you run evaluate method? If you did "
                "make sure scores are updated in evaluate method."
            )

        return self.scores


    def get_stats(self):
        """
        Returns:
            dict: stats that summarize the scores for each source using mean, median, \
                standard deviation, minimum, and maximum.
        """

        if not len(self.stats):
            self._compute_stats()

        return self.stats


    def save_scores_json(self, path):
        """
        Save scores to a JSON file

        Args:
            path (str): location to save JSON file
        """

        with open(path, 'w') as fp:
            json.dump(self.scores, fp, indent=True, cls=NumpyNumberEncoder)


    def save_stats_json(self, path):
        """
        Save score statistics as a JSON file

        Args:
            path (str): location to save JSON file
        """

        # Try to compute stats if it looks like they haven't been
        if not len(self.stats):
            self._compute_stats()

        # If there are now stats, save those
        if len(self.stats):
            with open(path, 'w') as fp:
                json.dump(self.stats, fp, indent=True, cls=NumpyNumberEncoder)

        else:
            print('No stats to save! Did you run evaluation?')


    def plot_hist(self, sources, metric, bins=None, clip_upper=None, **kwargs):
        """
        Plot a histogram of results of evaluation. Uses Matplotlib.

        Args:
            sources (list): Which audio sources to include in histogram. [0]
                would use the first prediction source passed in during construction,
                [1] would use the seconds, etc.
            metric (str): Which metric to use for creating the histogram. Depends on
                which were used during evaluation.
            bins (int or sequence or str, optional): passed into matplotlib hist
                method and indicates the number of bins to use, or if it is a list
                then it dfines bin edges. With Numpy 1.11 or newer, you can alternatively
                provide a string describing a binning strategy, such as 'auto',
                'sturges', 'fd', 'doane', 'scott', 'rice' or 'sqrt'
            clipper_upper (number, optional): Set an upper range for input values.
                This can be used to force any values above a certain range into the
                right most hitogram bin.
            kwargs: Keyword arguments to be passed into
                `hist <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.hist.html>`_
        """

        values = []
        for source in sources:
            values.extend([self.scores[key]['source_%s' % source][metric]
                           for key in self.scores])

        values = np.array(values)

        if clip_upper:
            values = np.clip(values, np.min(values), clip_upper)

        plt.hist(np.array(values), bins,
                 alpha=0.9, edgecolor='black', **kwargs)


    def verify_input_list(self, input_list):
        """
        Base method for verifying input list. Override to implement verification.
        For example, override and call
        :func:`~spiegelib.evaluation.EvaluationBase.verify_audio_input_list`
        on input_list to verify that :ref:`AudioBuffer <audio_buffer>` objects are being
        passed in.
        """


    @staticmethod
    def verify_audio_input_list(input_list):
        """
        Static method for verifying input lists with audio buffers
        """

        for item in input_list:
            if not isinstance(item, AudioBuffer):
                raise TypeError('Must be an AudioBuffer, received %s' \
                                % type(item))


    @staticmethod
    def euclidian_distance(A, B):
        """
        Calculates the euclidian distance between two arrays.

        Args:
            A (np.ndarray): First array  (Ground truth)
            B (np.ndarray): Second array (Prediction)

        Returns:
            float: Euclidian distance
        """
        return np.linalg.norm(np.subtract(A,B).flatten())


    @staticmethod
    def manhattan_distance(A, B):
        """
        Calculates the manhattan distance between two arrays.

        Args:
            A (np.ndarray): First array  (Ground truth)
            B (np.ndarray): Second array (Prediction)

        Returns:
            float: Manhattan distance
        """
        return np.linalg.norm(np.subtract(A,B).flatten(), 1)


    @staticmethod
    def mean_abs_error(A, B):
        """
        Calculates mean absolute error between two arrays. Mean(ABS(A-B)).

        Args:
            A (np.ndarray): First array  (Ground truth)
            B (np.ndarray): Second array (Prediction)

        Returns:
            float: Mean absolute error
        """
        return np.mean(np.abs(A-B).flatten())


    @staticmethod
    def mean_squared_error(A, B):
        """
        Calculates mean squared error between two arrays. Mean(Square(A-B)).

        Args:
            A (np.ndarray): First array  (Ground truth)
            B (np.ndarray): Second array (Prediction)

        Returns:
            float: Mean squared error
        """
        return np.mean(np.square(A-B).flatten())


    def _scores_to_nparray(self):
        """
        Reorganize scores into a set of np.ndarrays, one for each metric for each
        source. Returned as a dictionary: dict[source][metric]
        """

        summaries = {}
        scores = self.get_scores()
        num_targets = len(scores)
        i = 0

        # Organize scores into arrays organized by the source and the metric
        for target in scores:
            for source in scores[target]:
                if not source in summaries:
                    summaries[source] = {}

                for metric in scores[target][source]:
                    if not metric in summaries[source]:
                        summaries[source][metric] = np.zeros(num_targets)

                    summaries[source][metric][i] = scores[target][source][metric]

            i = i + 1

        return summaries


    def _compute_stats(self):
        """
        :returns: A dictionary of scores summarized using mean, median, and stddev
            for each source
        :rtype: dict
        """

        summaries = self._scores_to_nparray()

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
