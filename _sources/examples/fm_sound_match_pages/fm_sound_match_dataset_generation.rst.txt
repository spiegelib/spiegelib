Dataset Generation
------------------

First step is to generate and save datasets for training and validating deep
learning models. Additionally, we create a small audio dataset for
evaluation.

Here, we load the Dexed VST, and set the note length and render length to be one second.
For this experiment we aren’t worried about the release of the sound,
but you can set the render length longer than the note length to capture
the release portion of a signal. Synthesizer parameters are loaded from
a JSON file that describes all the overridden parameters and their
values.

.. code:: ipython3

    import spiegel

.. code:: ipython3

    synth = spiegel.synth.SynthVST("/Library/Audio/Plug-Ins/VST/Dexed.vst",
                                   note_length_secs=1.0, render_length_secs=1.0)
    synth.load_state("./synth_params/dexed_simple_fm.json")

MFCC Dataset
^^^^^^^^^^^^

Generate training and testing dataset using Mel-frequency Cepstral
Coefficients feature extraction. The DatasetGenerator class works by
generating random patches from the synthesizer, then running audio
feature extraction on the resulting sound, and then saving the audio
features and parameter values. Audio features and parameter values are saved in
seperate .npy files.

We set the `time_major` argument to `True` so that the orientation of the output
is (time_slices, features), as opposed to (features, time_slices) which
is default. This is how TensorFlow models expect the data to be
oriented.

Normalization settings used for the training dataset are saved as a .pkl file.
These settings are used to ensure future data is normalized in the same way.

The total size of this dataset is about 140MB.

.. code:: ipython3

    # Mel-frequency Cepstral Coefficients audio feature extractor.
    features = spiegel.features.MFCC(num_mfccs=13, time_major=True, hop_size=1024)

    # Setup generator for MFCC output and generate 50000 training examples and 10000 testing examples
    generator = spiegel.DatasetGenerator(synth, features,
                                         output_folder="./data_simple_FM_mfcc",
                                         normalize=True)
    generator.generate(50000, file_prefix="train_")
    generator.generate(10000, file_prefix="test_")
    generator.save_normalizers('normalizers.pkl')


STFT Dataset
^^^^^^^^^^^^

Generate training and testing dataset using the magnitude of the STFT.
This dataset will be used to train the convolutional neural network.

The total size of the resulting dataset is about 10.8GB.

.. code:: ipython3

    # Magnitude STFT ouptut feature extraction
    features = spiegel.features.STFT(fft_size=512, hop_size=256, output='magnitude', time_major=True)

    # Setup generator and create dataset
    generator = spiegel.DatasetGenerator(synth, features, output_folder="./data_simple_FM_stft", normalize=True)
    generator.generate(50000, file_prefix="train_")
    generator.generate(10000, file_prefix="test_")
    generator.save_normalizers('normalizers.pkl')


Evaluation Dataset
^^^^^^^^^^^^^^^^^^

Create an audio set for evaluation. We set the save_audio argument to
True in the DatasetGenerator constructor so that audio WAV files are
saved for this set.

.. code:: ipython3

    eval_generator = spiegel.DatasetGenerator(synth, features,
                                              output_folder='./evaluation',
                                              save_audio=True)
    eval_generator.generate(25)
