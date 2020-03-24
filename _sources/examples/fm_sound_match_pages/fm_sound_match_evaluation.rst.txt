Evaluation
----------

Finally, we evaluate the results of sound matching from both the deep
learning and genetic algorithms tested.

The MFCCEval class is used to perform an objective evaluation of each audio file generated. This is
carried out by measuring error metrics between the MFCCs from the target
sounds and the results from sound matching.

The results of this evaluation are saved in a JSON file which summarizes
the results for each estimator.

Histogram plots are also used to show the distribution of results
produced by each estimator. Histogram plots show the mean absolute error
of the sound matched results from each estimator.

.. code:: ipython3

   import spiegelib as spgl
   import numpy as np
   import matplotlib.pyplot as plt

.. code:: ipython3

  # Load the sound targets used for sound matching
  targets = spgl.AudioBuffer.load_folder('./evaluation/audio')

  # Load all the estimations of the sound targets made by each estimator
  estimations = [spgl.AudioBuffer.load_folder('./evaluation/mlp'),
                 spgl.AudioBuffer.load_folder('./evaluation/lstm'),
                 spgl.AudioBuffer.load_folder('./evaluation/bi_lstm'),
                 spgl.AudioBuffer.load_folder('./evaluation/cnn'),
                 spgl.AudioBuffer.load_folder('./evaluation/ga'),
                 spgl.AudioBuffer.load_folder('./evaluation/nsga')]

  # Evaluate the results and save to JSON file
  evaluation = spgl.evaluation.MFCCEval(targets, estimations)
  evaluation.evaluate()
  evaluation.save_stats_json('./evaluation/evaluation_stats.json')

MLP Histogram
^^^^^^^^^^^^^

.. code:: ipython3

  bins = np.arange(0, 40, 2.5)
  evaluation.plot_hist([0], 'mean_abs_error', bins)




.. image:: images/mlp_hist.png


LSTM Histogram
^^^^^^^^^^^^^^

.. code:: ipython3

  evaluation.plot_hist([1], 'mean_abs_error', bins)



.. image:: images/lstm_hist.png


LSTM++ Histogram
^^^^^^^^^^^^^^^^

.. code:: ipython3

  evaluation.plot_hist([2], 'mean_abs_error', bins)



.. image:: images/blstm_hist.png


CNN Histogram
^^^^^^^^^^^^^

.. code:: ipython3

  evaluation.plot_hist([3], 'mean_abs_error', bins)



.. image:: images/cnn_hist.png


GA Histogram
^^^^^^^^^^^^

.. code:: ipython3

  evaluation.plot_hist([4], 'mean_abs_error', bins)



.. image:: images/ga_hist.png


NSGA III Histogram
^^^^^^^^^^^^^^^^^^

.. code:: ipython3

  evaluation.plot_hist([5], 'mean_abs_error', bins)



.. image:: images/nsga_hist.png
