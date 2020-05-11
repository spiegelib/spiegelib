Audio Feature Extraction
========================

The following classes can be used to extract relevant features from audio. Audio
feature extraction helps to transform raw audio data into a representation that is
more representative of human perception. This representation is often used to provide
a more meaningful comparison of audio files and can be used as input to machine learning
approaches and to evaluate results.

All audio features in `SpiegeLib` currently use `librosa <https://librosa.github.io/>`_.
The :ref:`FeaturesBase <features_base>` class is an abstract base class that defines an interface
for performing feature extraction within `SpiegeLib`. All feature extraction classes
inherit from :ref:`FeaturesBase <features_base>` and provide a wrapper to functions within
`librosa <https://librosa.github.io/>`_.


.. toctree::
   :maxdepth: 2

	FeaturesBase <features/features_base>
   Fast Fourier Transform (FFT) <features/fft>
   Short Time Fourier Transform (STFT) <features/stft>
	Mel-Frequency Cepstral Coefficients (MFCC) <features/mfcc>
   Time Summarized Spectral Features (SpectralSummarized) <features/spectral_summarized>


Data Scaling:
"""""""""""""

Data scaling is an important step in pre-processing data prior to machine learning
algorithms. All classes that inherit from :ref:`FeaturesBase <features_base>` have
a ``scaler`` attribute which holds a data scaler object and can be used to normalize
or standardize feature extraction results. These scalers are inspired by the scalers
implemented in `sklearn <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing>`,
but can handle datasets with more dimensions. They are simplified and are designed
to be fit in one pass and to handle dense matrixes.

Scalers must be 'trained' (fit) before they can be used to scale new results. See the example
in :ref:`StandardScaler <standard_scaler>` for an example on how to perform this.

.. toctree::
   :maxdepth: 2

   DataScalerBase <features/data_scaler_base>
   StandardScaler <features/standard_scaler>
