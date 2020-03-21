Sound Match Deep Learning Models
--------------------------------

Now we can perform sound matching of the evaluation target set using the
trained deep learning models and save the resulting audio files to disk
for evaluation

.. code:: ipython3

    import spiegel

.. code:: ipython3

    # Load all saved models
    mlp = spiegel.estimator.TFEstimatorBase.load('./saved_models/simple_fm_mlp.h5')
    lstm = spiegel.estimator.TFEstimatorBase.load('./saved_models/simple_fm_lstm.h5')
    bi_lstm = spiegel.estimator.TFEstimatorBase.load('./saved_models/simple_fm_bi_lstm.h5')
    cnn = spiegel.estimator.TFEstimatorBase.load('./saved_models/simple_fm_cnn.h5')

.. code:: ipython3

    # Load synth with overriden params
    synth = spiegel.synth.SynthVST("/Library/Audio/Plug-Ins/VST/Dexed.vst",
                                   note_length_secs=1.0, render_length_secs=1.0)
    synth.load_state("./synth_params/dexed_simple_fm.json")

Setup all the feature extractors to provide the correct input data for
each model based on how it was trained. Also use the same data
normalizers that were setup when initially creating each dataset.

.. code:: ipython3

    # MLP feature extractor with a modifying function that flattens the time slice arrays at the end of the feature
    # extraction pipeline
    mlp_extractor = spiegel.features.MFCC(num_mfccs=13, time_major=True, hop_size=1024, normalize=True)
    mlp_extractor.load_normalizers('./data_simple_fm_mfcc/normalizers.pkl')
    mlp_extractor.add_modifier(lambda data : data.flatten(), type='output')

    # LSTM & LSTM++ feature extractor -- time series of MFCC frames
    lstm_extractor = spiegel.features.MFCC(num_mfccs=13, time_major=True, hop_size=1024, normalize=True)
    lstm_extractor.load_normalizers('./data_simple_fm_mfcc/normalizers.pkl')

    # CNN feature extractor uses magnitude output from STFT and then modifies the output array into a 3D array for the
    # 2D convolutional network becuase it is expecting an image with a single channel (ie grayscale).
    cnn_extractor = spiegel.features.MagnitudeSTFT(fft_size=512, hop_size=256, time_major=True, normalize=True)
    cnn_extractor.load_normalizers('./data_simple_fm_stft/normalizers.pkl')
    cnn_extractor.add_modifier(lambda data : data.reshape(data.shape[0], data.shape[1], 1), type='output')

SoundMatch is a class designed to help run sound matches for a
synthesizer and a specific estimator type. Each SoundMatch object
requires a synthesizer to use to generate sounds, an estimator object,
and optionally an audio feature extractor object. If an audio feature
object is provided, that will be used to extract features from incoming
audio prior to running estimation. This is required for these deep
learning models, but some estimators can handle raw audio, such as the
genetic estimators.

.. code:: ipython3

    mlp_matcher = spiegel.SoundMatch(synth, mlp, mlp_extractor)
    lstm_matcher = spiegel.SoundMatch(synth, lstm, lstm_extractor)
    bi_lstm_matcher = spiegel.SoundMatch(synth, bi_lstm, lstm_extractor)
    cnn_matcher = spiegel.SoundMatch(synth, cnn, cnn_extractor)

Load in the folder of evaluation audio samples and perform sound
matching on each one with each estimation model. AudioBuffer.load_folder
performs a natural sort based on the file names of the audio contained
in the specified folder, so we can save each prediction with a
corresponding integer number and be assured that the ordering will match
up when we get to evaluation.

.. code:: ipython3

    targets = spiegel.AudioBuffer.load_folder('./evaluation/audio')

    for i in range(len(targets)):
        audio = mlp_matcher.match(targets[i])
        audio.save('./evaluation/mlp/mlp_prediction_%s.wav' % i)

        audio = lstm_matcher.match(targets[i])
        audio.save('./evaluation/lstm/lstm_prediction_%s.wav' % i)

        audio = bi_lstm_matcher.match(targets[i])
        audio.save('./evaluation/bi_lstm/bi_lstm_prediction_%s.wav' % i)

        audio = cnn_matcher.match(targets[i])
        audio.save('./evaluation/cnn/cnn_prediction_%s.wav' % i)
