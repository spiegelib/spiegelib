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

    Ex) targetList = [y1, y2]
        estimatedList = [[y1_estimation1, y1_estimation2], [y2_estimation1, y2_estimation2]]

    :param targetList: list of :class:`spiegel.core.audio_buffer.AudioBuffer` objects
        to use as the ground truth for evaluation.
    :type targetList: list
    :param estimatedList: List of lists of :class:`spiegel.core.audio_buffer.AudioBuffer` objects.
        There must be as many lists as there are targets, and each of those lists contain
        AudioBuffers that are estimations for the associated target AudioBuffer.
    :type estimatedList: list
    :param kwargs: keyword arguments to pass to base class. See :class:`spiegel.evaluation.audio_eval_base.AudioEvalBase`
    """

    def __init__(self, targetList, estimatedList, sampleRate=None, **kwargs):
        """
        Constructor
        """
        self.sampleRate = sampleRate if sampleRate else targetList[0].getSampleRate()
        super().__init__(targetList, estimatedList, **kwargs)


    def evaluateTarget(self, target, predictions):
        """
        Evaluate absolute mean error and mean squared error between MFCCs of all
        target AudioBuffers and estimated AudioBuffers.
        :returns: A list of metric dictionaries for each prediction
        :rtype: list
        """

        results = []
        mfcc = MFCC(sampleRate=self.sampleRate)
        targetMFCCs = mfcc(target)

        for pred in predictions:
            estimatedMFCCs = mfcc(pred)
            results.append({
                'absoluteMeanError': EvaluationBase.absoluteMeanError(targetMFCCs, estimatedMFCCs),
                'meanSquaredError': EvaluationBase.meanSquaredError(targetMFCCs, estimatedMFCCs),
                'euclidianDistance': EvaluationBase.euclidianDistance(targetMFCCs, estimatedMFCCs),
                'manhattanDistance': EvaluationBase.manhattanDistance(targetMFCCs, estimatedMFCCs),
            })

        return results


    def verifyInputList(self, inputList):
        """
        Overriding verification method to check for AudioBuffers
        """
        EvaluationBase.verifyAudioInputList(inputList)
