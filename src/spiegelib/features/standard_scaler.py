#!/usr/bin/env python
"""
Data scaler class that implments standarization. This transforms data
to have a mean of zero and variance of 1.

Scaler must be 'trained' (fit) before it can be used to scale new results.

Examples:

    1) Feature extraction objects by default load with a StandardScaler and
       are designed to work with the DatasetGenetor

        .. code:: ipython3

            import spiegelib as spgl

            synth = spgl.synth.SynthVST("/Library/Audio/Plug-Ins/VST/Dexed.vst")
            spectral_features = spgl.features.SpectralSummarized()
            generator = spgl.DatasetGenerator(synth, spectral_features, scale=True)
            generator.generate(1000)
            generator.save_scaler('./data_scaler.pkl')

        The scaler is saved in the spectral_features object and is used to scale
        future data. The scaler is also saved and can be reloaeded for future
        feature extraction. See :py:meth:`~spiegelib.features.FeaturesBase.load_scaler`



    2) Using StandardScaler independently

        .. code:: ipython3

            import spiegelib as spgl
            import numpy as np

            # Generate a dataset without scaling
            synth = spgl.synth.SynthVST("/Library/Audio/Plug-Ins/VST/Dexed.vst",
                                        render_length_secs=1.0)

            mfcc = spgl.features.MFCC(num_mfccs=13)
            generator = spgl.DatasetGenerator(synth, mfcc)
            generator.generate(1000, file_prefix="train_")


            # DatasetGenerator will automatically save the extracted features as a npy
            # file in the current working directory
            dataset = np.load('./train_features.npy')

            scaler = spgl.features.StandardScaler()
            scaler.fit(dataset)
            scaled_data = scaler.transform(dataset)


            # Now we can add the scaler to the MFCC feature extractor and use it to
            # scale any future feature extractions
            mfcc.set_scaler(scaler)
            random_audio = synth.get_random_example()
            scaled_mfccs = mfcc(random_audio, scale=True)

    3) Scaling along certain dimensions

        The dimension that scaling is applied to depends on the fit axis. Our MFCC dataset
        generated in the previous example has the shape (1000, 13, 88) where each axis
        corresponds to (batches, mfccs, time slices).

        .. code:: ipython3

            # Same dataset as before and new scaler object
            dataset = np.load('./train_features.npy')
            scaler = spgl.features.StandardScaler()

            # This flattens the entire dataset and calculates
            # the mean and variance on the flattened array
            scaler.fit(dataset)

            # Since the batch is on the first axis, this will calculate
            # the mean and variance independently for each MFCC and time slice
            scaler.fit(dataset, axis=0)

            # This will scale each MFCC band independently
            scaler.fit(dataset, axis=(0,2))

        To control the scale axis when using the DatasetGenerator from the first
        example, a custom axis can be set for a feature extraction object in the
        constructor. See the scale_axis argument in
        the :py:class:`~spiegelib.features.FeaturesBase` constructor.
"""
import numpy as np
from spiegelib.features import DataScalerBase

class StandardScaler(DataScalerBase):

    def _reset(self):
        """
        Reset attributes
        """

        if hasattr(self, 'mean'):
            del self.mean
            del self.std


    def fit(self, data, axis=None):
        """
        Compute mean and std for later scaling

        Args:
            data (np.ndarray): data to calculate mean and standard deviation on
                for future scaling
            axis (int, tuple, optional): axis or axes to use for calculating scaling
                parameteres. Defaults to None which will flatten the array first.
        """

        self._reset()
        self.mean = data.mean(axis)

        # Calculate standard deviation and handle zeros
        variance = data.var(axis)
        if isinstance(variance, np.ndarray):
            variance[variance == 0.0] = 1.0
            self.std = np.sqrt(variance)

        elif np.isscalar(variance):
            variance = 1.0 if variance == 0.0 else variance
            self.std = np.sqrt(variance)

        self.fit_axis = axis
        if self.fit_axis != None and not isinstance(self.fit_axis, tuple):
            self.fit_axis = (self.fit_axis,)
        self.fit_shape = data.shape


    def transform(self, data):
        """
        Scale data

        Args:
            data (np.ndarray): data to scale

        Returns:
            np.ndarray: scaled data
        """

        if not hasattr(self, 'mean'):
            raise Exception("You must fit this scaler first")

        if self.fit_axis is None:
            return (data - self.mean) / self.std

        mean_expanded = np.copy(self.mean)
        std_expanded = np.copy(self.std)

        if len(data.shape) == len(self.fit_shape):
            for axis in self.fit_axis:
                mean_expanded = np.expand_dims(mean_expanded, axis=axis)
                std_expanded = np.expand_dims(std_expanded, axis=axis)

        elif len(data.shape) == len(self.fit_shape) - 1:
            for axis in self.fit_axis[1:]:
                mean_expanded = np.expand_dims(mean_expanded, axis=(axis-1))
                std_expanded = np.expand_dims(std_expanded, axis=(axis-1))

        else:
            raise ValueError("Input data has incorrect shape. Expected %s or %s, received %s"
                             % (len(self.fit_shape), len(self.fit_shape) -1, len(data.shape)))

        return (data - mean_expanded) / std_expanded
