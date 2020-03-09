FM Sound Match Experiment
=========================

.. code:: ipython3

    import spiegel

Load Dexed VST, set the note length and render length to be one second.
For this experiment we aren’t worried about the release of the sound,
but you can set the render length longer than the note length to capture
the release portion of a signal. Synthesizer parameters are loaded from
a JSON file that describes all the overrriden parameters and their
values.

.. code:: ipython3

    synth = spiegel.synth.SynthVST("/Library/Audio/Plug-Ins/VST/Dexed.vst", note_length_secs=1.0, render_length_secs=1.0)
    synth.load_state("./synth_params/dexed_simple_fm.json")

.. code:: ipython3

    features = spiegel.features.MFCC(num_mfccs=13, time_major=True, hop_size=1024)
