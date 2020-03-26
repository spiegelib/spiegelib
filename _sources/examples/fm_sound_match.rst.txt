FM Sound Match Experiment
=========================

This experiment compares several different algorithms for estimating parameters for
an FM synthesizer. The goal is to be able to select synthesizer parameters in order
to replicate a target sound as closely as possible. This is called sound matching. We'll
run this experiment using the open-source *Dexed* VST emulation of the Yamaha DX7.
Dexed can be dowloaded for free `here <https://asb2m10.github.io/dexed/>`_.

Through this example we will use *SpiegeLib* to:

* Program and generate sounds from a VST synthesizer
* Generate datasets for deep learning and evaluation
* Train deep learning models
* Perform sound matching using deep learning and genetic algorithms
* Evaluate results

If you haven't already, make sure you have *SpiegeLib* and *RenderMan* installed.
See :ref:`installation instructions <installation>`. And download `Dexed <https://asb2m10.github.io/dexed/>`__.

All code is available as Python notebooks on the project `github page <https://github.com/spiegelib/vst-fm-sound-match>`__.
The trained models from this experiment are also
available in the git repo. All datasets generated and used in
this experiment are also available online.

Experiment Sections
^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   Synthesizer Configuration <fm_sound_match_pages/fm_sound_match_synth_config>
   Dataset Generation <fm_sound_match_pages/fm_sound_match_dataset_generation>
   Train Deep Learning Models <fm_sound_match_pages/fm_sound_match_train_models>
   Sound Match Deep Learning Models <fm_sound_match_pages/fm_sound_match_deep_learning>
   Sound Match Genetic Algorithms <fm_sound_match_pages/fm_sound_match_genetic>
   Evaluation <fm_sound_match_pages/fm_sound_match_evaluation>
