#!/use/bin/env python
"""
Audio evaluation using MFCC error between targets and estimations
"""

from spiegel.evaluation.audio_eval_base import AudioEvalBase
from spiegel.features.mfcc import MFCC


class MFCCEval(AudioEvalBase):
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

    def __init__(self, targetList, estimatedList, **kwargs):
        """
        Constructor
        """
        super().__init__(targetList, estimatedList, **kwargs)


    def evaluate(self):
        """
        Evaluate absolute mean error and mean squared error between MFCCs of all
        target AudioBuffers and estimated AudioBuffers. Stores results as a dictionary
        in a member function that can be accessed through getScores() member function.
        """

        results = {}
        mfcc = MFCC(sampleRate=self.sampleRate)

        for i in range(len(self.targetList)):

            targetMFCCs = mfcc.getFeatures(self.targetList[i])
            targetResults = {}

            for j in range(len(self.estimatedList[i])):
                estimatedMFCCs = mfcc.getFeatures(self.estimatedList[i][j])
                targetResults['estimation_%s' % j] = {
                    'absoluteMeanError': AudioEvalBase.absoluteMeanError(targetMFCCs, estimatedMFCCs),
                    'meanSquaredError': AudioEvalBase.meanSquaredError(targetMFCCs, estimatedMFCCs),
                }

            results['target_%s' % i] = targetResults

        self.scores = results
