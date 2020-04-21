#!/use/bin/env python
"""
Audio evaluation using MFCC error between targets and estimations. See
:class:`~spiegelib.evaluation.EvaluationBase` for documentation of methods for
running objective evaluation using inheriting classes, such as this one.

Example::

    import spiegelib as spgl

    targets = spgl.AudioBufer('./folder_of_target_audio_files')
    ga_predictions = spgl.AudioBuffer('./folder_of_ga_predictions')
    mlp_predictions = spgl.AudioBuffer('./folder_of_mlp_predictions')
    estimations = [ga_predictions, mlp_predictions]

    # Instantiate evaluation with audio buffers and run evaluation
    evaluation = spgl.evaluation.MFCCEval(targets, estimations)
    evaluation.evaluate()

    # Save raw results and stats of results as JSON
    evaluation.save_scores_json('./mfcc_results.json')
    evaluation.save_stats_json('./mfcc_eval_stats.json')

    # Plot a histogram of GA predictions using mean absolute error metrics
    evaluation.plot_hist([0], 'mean_abs_error')

    # Plot a histogram of MLP predictions using mean absulote error metrics
    evaluation.plot_hist([1], 'mean_abs_error')
"""

from spiegelib.evaluation.evaluation_base import EvaluationBase
from spiegelib.features.mfcc import MFCC


class MFCCEval(EvaluationBase):
    """
    Pass in a list of target :class:`~spiegelib.AudioBuffer` objects and lists of
    estimated :class:`~spiegelib.AudioBuffer` objects for each target.

    Example Input::

        targets = [target_1, target_2]
        estimations = [[prediction_1_for_target_1, prediction_2_for_target_1],
                       [prediction_1_for_target_2, prediction_2_for_target_2]]

    Where prediction_1 and prediction_2 would represent results from two different
    methods or sources. For example, audio for all prediction_1 samples might be
    from a GA and all audio for prediction_2 samples might be from a deep learning
    approach.

    Args:
        targets (list): a list of :ref:`AudioBuffer <audio_buffer>` objects to use
            as the ground truth for evaluation
        estimations (lits): a 2D list of :ref:`AudioBuffer <audio_buffer>` objects.
            Should contain a list of AudioBuffers for each target. The position of an
            AudioBuffer in each list is used to distinguish between different
            audio sources. For example, if you are comparing two different
            synthesizer programming methods, then you would want to make sure to
            have the results from each method in the same position in each list.
        sample_rate (int, optional): sample rate to run feature extraction at. Defaults
            to the audio rate of the first target.
        kwargs: Keyword arguments to pass into :class:`~spiegelib.evaluation.EvaluationBase`
    """

    def __init__(self, targets, estimations, sample_rate=None, **kwargs):
        """
        Constructor
        """
        self.sample_rate = sample_rate if sample_rate else targets[0].get_sample_rate()
        super().__init__(targets, estimations, **kwargs)


    def evaluate_target(self, target, predictions):
        """
        Called automatically by :func:`spiegelib.evaluation.EvaluationBase.evaluate`

        Evaluates difference between a target :ref:`AudioBuffer <audio_buffer>` and
        a list of estimation :ref:`AudioBuffers <audio_buffer>`. Calculates mean absolute
        error, mean squared error, euclidian distance, and manhattan distance on results
        and returns a dictionary for each estimation with results in a list.

        Args:
            target (list): Audio to use as ground truth in evaluation
            predictions (list): list of :ref:`AudioBuffer <audio_buffer>` objects to evaluate
                against the target audio file.
        Returns:
            list: A list of dictionaries with stats for each prediction evaluation
        """

        results = []
        mfcc = MFCC(sample_rate=self.sample_rate)
        target_mfccs = mfcc(target)

        for pred in predictions:
            estimated_mfccs = mfcc(pred)
            results.append({
                'mean_abs_error': EvaluationBase.mean_abs_error(target_mfccs, estimated_mfccs),
                'mean_squared_error': EvaluationBase.mean_squared_error(target_mfccs, estimated_mfccs),
                'euclidian_distance': EvaluationBase.euclidian_distance(target_mfccs, estimated_mfccs),
                'manhattan_distance': EvaluationBase.manhattan_distance(target_mfccs, estimated_mfccs),
            })

        return results


    def verify_input_list(self, input_list):
        """
        Overriding verification method to check for AudioBuffers. Called automatically
        during construction.
        """
        EvaluationBase.verify_audio_input_list(input_list)
