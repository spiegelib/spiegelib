Synthesizer Configuration
-------------------------

First, we setup Dexed for the experiment by selecting
parameters to automatically program. To simplify the experiment, we’ll
only use a small subset of all the available parameters.

.. code:: ipython3

    import spiegelib as spgl

Load Dexed into an instance of SynthVST

.. code:: ipython3

    synth = spgl.synth.SynthVST("/Library/Audio/Plug-Ins/VST/Dexed.vst")

Dexed is an emulation of the Yamaha DX7 which contains six different
operators that can be arranged in various ways to create complex sounds.
Dexed has 155 parameters for controlling these operators and other
global options and FX. In order to simplify this experiment we’re going
to focus on a small subset of these parameters and treat Dexed as a
simple two-operator FM synthesizer. In order to do that, we will
override and freeze most of the other parameters.

To start, we’ll save the parameter values to a JSON file showing the
parameter ID, values, and a short description of all the available
parameters. Parameters can also be flagged as overridden in the JSON
file. This allows parameter configurations to be saved, modified, and reloaded.

.. code:: ipython3

    synth.save_state("./synth_params/dexed_simple_fm_init.json")

We can now manually edit the JSON file and reload it to update our
synthesizer configuration. We can also programmatically set overridden
parameters and save a new synthesizer configuration file. We’ll do the
latter.

All parameters have a range between 0.0 and 1.0

.. code:: ipython3

    # The algorithm paramter sets the arrangement of the FM operators.
    # There are 32 differen arrangements available in Dexed.
    # This selects the first algorithm. This algorithm sets
    # the second operator to frequency modulate the first.
    algorithm_number = 1
    alg = (1.0 / 32.0) * float(algorithm_number - 1) + 0.001

    overridden_parameters = [
        (0, 1.0), # Filter Cutoff (Fully open)
        (1, 0.0), # Filter Resonance
        (2, 1.0), # Output Gain
        (3, 0.5), # Master Tuning (Center is 0)
        (4, alg), # Operator configuration
        (5, 0.0), # Feedback
        (6, 1.0), # Key Sync Oscillators
        (7, 0.0), # LFO Speed
        (8, 0.0), # LFO Delay
        (9, 0.0), # LFO Pitch Modulation Depth
        (10, 0.0),# LFO Amplitude Modulation Depth
        (11, 0.0),# LFO Key Sync
        (12, 0.0),# LFO Waveform
        (13, 0.5),# Middle C Tuning
    ]

    # Turn off all pitch modulation parameters
    overridden_parameters.extend([(i, 0.0) for i in range(14, 23)])

    # Turn Operator 1 into a simple sine wave with no envelope
    overridden_parameters.extend([
        (23, 0.9), # Operator 1 Attack Rate
        (24, 0.9), # Operator 1 Decay Rate
        (25, 0.9), # Operator 1 Sustain Rate
        (26, 0.9), # Operator 1 Release Rate
        (27, 1.0), # Operator 1 Attack Level
        (28, 1.0), # Operator 1 Decay Level
        (29, 1.0), # Operator 1 Sustain Level
        (30, 0.0), # Operator 1 Release Level
        (31, 1.0), # Operator 1 Gain
        (32, 0.0), # Operator 1 Mode (1.0 is Fixed Frequency)
        (33, 0.5), # Operator 1 Coarse Tuning
        (34, 0.0), # Operator 1 Fine Tuning
        (35, 0.5), # Operator 1 Detune
        (36, 0.0), # Operator 1 Env Scaling Param
        (37, 0.0), # Operator 1 Env Scaling Param
        (38, 0.0), # Operator 1 Env Scaling Param
        (39, 0.0), # Operator 1 Env Scaling Param
        (40, 0.0), # Operator 1 Env Scaling Param
        (41, 0.0), # Operator 1 Env Scaling Param
        (42, 0.0), # Operator 1 Mod Sensitivity
        (43, 0.0), # Operator 1 Key Velocity
        (44, 1.0), # Operator 1 On/Off switch
    ])

    # Override some of Operator 2 parameters
    overridden_parameters.extend([
        (45, 0.9), # Operator 2 Attack Rate (No attack on operator 2)
        (49, 1.0), # Operator 2 Attack Level
        (53, 1.0), # Operator 2 Gain (Operator 2 always outputs)
        (54, 0.0), # Operator 1 Mode (1.0 is Fixed Frequency)
        (58, 0.0), # Operator 1 Env Scaling Param
        (59, 0.0), # Operator 1 Env Scaling Param
        (60, 0.0), # Operator 1 Env Scaling Param
        (61, 0.0), # Operator 1 Env Scaling Param
        (62, 0.0), # Operator 1 Env Scaling Param
        (63, 0.0), # Operator 1 Env Scaling Param
        (64, 0.0), # Operator 1 Mod Sensitivity
        (65, 0.0), # Operator 1 Key Velocity
        (66, 1.0), # Operator 1 On/Off switch
    ])

    # Override operators 3 through 6
    overridden_parameters.extend([(i, 0.0) for i in range(67, 155)])

    # Set overridden parameters in synth
    synth.set_overridden_parameters(overridden_parameters)


Now we have a synthesizer configuration setup that turns Dexed into a
simple two-operator FM synthesizer. Only nine parameters have been left
un-overridden. These parameters control the pitch and envelope
parameters of operator two. Let’s save that configuration so we can
reuse it throughout this experiment.

.. code:: ipython3

    synth.save_state("./synth_params/dexed_simple_fm.json")
