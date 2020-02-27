#!/use/bin/env python
"""
Audio evaluation using MFCC error between targets and estimations
"""

from spiegel.evaluation.evaluation_base import EvaluationBase
from spiegel.features.mfcc import MFCC


class MFCCEval(EvaluationBase):
    """
    Pass in a list of target AudioBuffers and lists of estimated AudioBuffers for
    each target AudioBuffer.

    Ex) targets = [y1, y2]
        estimations = [[y1_estimation1, y1_estimation2], [y2_estimation1, y2_estimation2]]

    :param targets: list of :class:`spiegel.core.audio_buffer.AudioBuffer` objects
        to use as the ground truth for evaluation.
    :type targets: list
    :param estimations: List of lists of :class:`spiegel.core.audio_buffer.AudioBuffer` objects.
        There must be as many lists as there are targets, and each of those lists contain
        AudioBuffers that are estimations for the associated target AudioBuffer.
    :type estimations: list
    :param kwargs: keyword arguments to pass to base class. See
        :class:`spiegel.evaluation.audio_eval_base.AudioEvalBase`
    """

    def __init__(self, targets, estimations, sample_rate=None, **kwargs):
        """
        Constructor
        """
        self.sample_rate = sample_rate if sample_rate else targets[0].get_sample_rate()
        super().__init__(targets, estimations, **kwargs)


    def evaluate_target(self, target, predictions):
        """
        Evaluate absolute mean error and mean squared error between MFCCs of all
        target AudioBuffers and estimated AudioBuffers.
        :returns: A list of metric dictionaries for each prediction
        :rtype: list
        """

        results = []
        mfcc = MFCC(sample_rate=self.sample_rate)
        target_mfccs = mfcc(target)

        for pred in predictions:
            estimated_mfccs = mfcc(pred)
            results.append({
                'abs_mean_error': EvaluationBase.abs_mean_error(target_mfccs, estimated_mfccs),
                'mean_squared_error': EvaluationBase.mean_squared_error(target_mfccs, estimated_mfccs),
                'euclidian_distance': EvaluationBase.euclidian_distance(target_mfccs, estimated_mfccs),
                'manhattan_distance': EvaluationBase.manhattan_distance(target_mfccs, estimated_mfccs),
            })

        return results


    def verify_input_list(self, input_list):
        """
        Overriding verification method to check for AudioBuffers
        """
        EvaluationBase.verify_audio_input_list(input_list)
