#!/usr/bin/env python
"""
Trained VGGish Feature Extractor
"""

import numpy as np
import tensorflow.compat.v1 as tf

from spiegelib import AudioBuffer
from spiegelib.features.features_base import FeaturesBase

import spiegelib.features.vggish.vggish_input as vggish_input
import spiegelib.features.vggish.vggish_params as vggish_params
import spiegelib.features.vggish.vggish_slim as vggish_slim

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from . import vggish_data

class VGGishExtractor(FeaturesBase):
    """
    Args:
        scale_axis (int, tuple, None): When applying scaling, determines which dimensions
            scaling be applied along. Defaults to None, which will flatten results and
            calculate scaling variables on that.
        kwargs: Keyword arguments, see :class:`spiegelib.features.features_base.FeaturesBase`.
    """

    def __init__(self, scale_axis=None, **kwargs):
        """
        Contructor
        """

        # Setup feature base class -- FFT is time summarized, so no
        # time slices are used, defaults to normalizing the entire result
        # as opposed to normalizing across each bin separately.
        super().__init__(scale_axis=scale_axis, **kwargs)

    def get_features(self, audio):
        """
        Run VGGish Extractor on audio

        Args:
            audio (:ref:`AudioBuffer <audio_buffer>`): input audio

        Returns:
            np.ndarray: Results of VGGish feature extraction
        """

        if not isinstance(audio, AudioBuffer):
            raise TypeError('audio must be AudioBuffer, recieved %s' % type(audio))

        examples_batch = vggish_input.waveform_to_examples(audio.get_audio(), audio.get_sample_rate())

        with tf.Graph().as_default(), tf.Session() as sess:
            # Define the model in inference mode, load the checkpoint, and
            # locate input and output tensors.
            vggish_slim.define_vggish_slim(training=False)

            with pkg_resources.path(vggish_data.__name__, 'vggish_model.ckpt') as checkpoint:
                vggish_slim.load_vggish_slim_checkpoint(sess, str(checkpoint.resolve()))

            features_tensor = sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
            embedding_tensor = sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)

            # Run inference and postprocessing.
            [embedding_batch] = sess.run([embedding_tensor],
                                         feed_dict={features_tensor: examples_batch})

        features = np.array(embedding_batch)
        return features
