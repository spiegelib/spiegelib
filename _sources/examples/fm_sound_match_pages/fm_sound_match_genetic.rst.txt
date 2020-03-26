
Sound Match Genetic Algorithm Estimators
----------------------------------------

Now we perform sound matching of the evaluation target set using
two genetic algorithm based approaches and save the resulting audio
files to disk for evaluation.

.. code:: ipython3

    import spiegelib as spgl

.. code:: ipython3

   # Load synth with overridden params
   synth = spgl.synth.SynthVST("/Library/Audio/Plug-Ins/VST/Dexed.vst",
                               note_length_secs=1.0, render_length_secs=1.0)
   synth.load_state("./synth_params/dexed_simple_fm.json")

Basic Genetic Algorithm
^^^^^^^^^^^^^^^^^^^^^^^

Setup the feature extractor for the basic single-objective genetic
algorithm. It uses a 13-band MFCC, which is calculated on every new
individual in the population. The error between an individual and the
target audio sound is used the evaluate the fitness of each individual.

.. code:: ipython3

   # MFCC features
   ga_extractor = spgl.features.MFCC(num_mfccs=13, hop_size=1024)

   # Basic Genetic Algorithm estimator
   ga = spgl.estimator.BasicGA(synth, ga_extractor, pop_size=300, ngen=100)

   # Sound matching helper class
   ga_matcher = spgl.SoundMatch(synth, ga)

Non-dominated sorting genetic algorithm III (NSGA III)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Setup the feature extractors for the multi-objective genetic algorithm.
Each extractor is used for one of the GA objectives, so in this instance
there are 3 objectives: MFCC, Spectral Summarized, and the FFT.

.. code:: ipython3

   # Feature extractors for Multi-Objective GA
   nsga_extractors = [spgl.features.MFCC(num_mfccs=13, hop_size=1024),
                      spgl.features.SpectralSummarized(hop_size=1024),
                      spgl.features.FFT(output='magnitude')]

   # NSGA3 Multi-Objective Genetic Algorithm
   nsga = spgl.estimator.NSGA3(synth, nsga_extractors)

   # Sound matching helper class
   nsga_matcher = spgl.SoundMatch(synth, nsga)

Sound Matching
^^^^^^^^^^^^^^

Load in the folder of evaluation audio samples and perform sound
matching on each one using both genetic algorithms. This may take
several hours to run on all 25 sounds.

.. code:: ipython3

   targets = spgl.AudioBuffer.load_folder('./evaluation/audio')

   for i in range(len(targets)):
       audio = ga_matcher.match(targets[i])
       audio.save('./evaluation/ga/ga_predicition_%s.wav' % i)

       audio = nsga_matcher.match(targets[i])
       audio.save('./evaluation/nsga/nsga_prediction_%s.wav' % i)
