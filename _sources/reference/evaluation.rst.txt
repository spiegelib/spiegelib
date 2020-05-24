Evaluation Classes
==================

The following classes can be used to evaluate the results of experiments both
objectively and subjectively.

:ref:`EvaluationBase <evaluation_base>` provides a common base class for evaluation
methods that compare a set of audio files or other objects to references or targets.
This abstract class also has a set of static methods for computing error and distance
metrics between np.ndarrays and also for calculating statistics on results of
evaluations.

Objective evaluation of audio files can be carried out using the :ref:`MFCCEval <mfcc_eval>`
class.

Simple browser based listening tests can be setup and hosted locally using the
:ref:`Subjective <subjective_evaluation>` class.

.. toctree::
   :maxdepth: 2

	EvaluationBase <evaluation/evaluation_base>
	MFCCEval <evaluation/mfcc_eval>
   Subjective <evaluation/subjective>
