"""
Tests for MinMaxScaler class
"""

import pytest
import numpy as np

from spiegelib.features import MinMaxScaler

class TestMinMaxScaler():

    def test_empty_construction(self):
        scaler = MinMaxScaler()


    def test_fitting_full(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        scaler = MinMaxScaler()
        scaler.fit(features)
        assert scaler.min == features.min()
        assert scaler.max == features.max()

    def test_fit_and_transform_full(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        scaler = MinMaxScaler()

        scaler.fit(features)
        assert scaler.min == pytest.approx(-638.7879)
        assert scaler.max == pytest.approx(306.32843)

        scaled = scaler.transform(features)
        assert scaled.min() == pytest.approx(0.0, abs=1e-9)
        assert scaled.max() == pytest.approx(1.0)

    def test_fit_transform_full(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        scaler = MinMaxScaler()

        scaled = scaler.fit_transform(features)
        assert scaler.min == pytest.approx(-638.7879)
        assert scaler.max == pytest.approx(306.32843)
        assert scaled.min() == pytest.approx(0.0, abs=1e-9)
        assert scaled.max() == pytest.approx(1.0)

    def test_transform_slice(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        scaler = MinMaxScaler()
        scaler.fit(features)

        scaled = scaler.transform(features[0])
        assert scaled.min() == pytest.approx(0.18291706)
        assert scaled.max() == pytest.approx(0.9821132)


    def test_fit_per_feature(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        scaler = MinMaxScaler()
        scaler.fit(features, (0,1))

        expected_min = np.load((shared_datadir / 'test_mfcc/per_feature_min.npy').resolve())
        np.testing.assert_array_almost_equal(scaler.min, expected_min)

        expected_max = np.load((shared_datadir / 'test_mfcc/per_feature_max.npy').resolve())
        np.testing.assert_array_almost_equal(scaler.max, expected_max)


    def test_fit_and_transform_per_feature(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        scaler = MinMaxScaler()
        scaler.fit(features, (0,1))
        scaled = scaler.transform(features)

        expected_min = np.zeros((features.shape[2],))
        np.testing.assert_array_almost_equal(scaled.min((0,1)), expected_min, 1e-5)

        expected_max = np.full(features.shape[2], 1)
        np.testing.assert_array_almost_equal(scaled.max((0,1)), expected_max, 1e-5)


    def test_transform_per_feature_slice(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        scaler = MinMaxScaler()
        scaler.fit(features, (0,1))
        scaled = scaler.transform(features[0])

        assert scaled.shape == features[0].shape
        assert scaled.min() == pytest.approx(0.09640828)
        assert scaled.max() == pytest.approx(0.93788314)


    def test_fit_and_transform_per_feature_non_time_major(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        features = np.swapaxes(features, 1, 2)[0:13]
        scaler = MinMaxScaler()
        scaler.fit(features, (0,2))
        scaled = scaler.transform(features)

        expected_min = np.zeros((features.shape[1],))
        np.testing.assert_array_almost_equal(scaled.min((0,2)), expected_min)

        expected_max = np.full(features.shape[1], 1)
        np.testing.assert_array_almost_equal(scaled.max((0,2)), expected_max)


    def test_transform_per_feature_slice_non_time_major(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        features = np.swapaxes(features, 1, 2)
        scaler = MinMaxScaler()
        scaler.fit(features, (0,2))
        scaled = scaler.transform(features[0])

        assert scaled.shape == features[0].shape
        assert scaled.min() == pytest.approx(0.09640828)
        assert scaled.max() == pytest.approx(0.93788314)


    def test_fit_time_summary_full(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        features = features.mean(1)
        scaler = MinMaxScaler()
        scaler.fit(features)

        expected_min = features.min()
        expected_max = features.max()
        assert scaler.min == pytest.approx(expected_min)
        assert scaler.max == pytest.approx(expected_max)


    def test_transform_time_summary_full(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        features = features.mean(1)
        scaler = MinMaxScaler()
        scaler.fit(features)

        scaled = scaler.transform(features)
        assert scaled.shape == features.shape
        assert scaled.min() == pytest.approx(0.0, abs=1e-6)
        assert scaled.max() == pytest.approx(1.0)


    def test_transform_time_summary_full_slice(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        features = features.mean(1)
        scaler = MinMaxScaler()
        scaler.fit(features)

        scaled = scaler.transform(features[0])
        assert scaled.shape == features[0].shape
        assert scaled.min() == pytest.approx(0.260660088, abs=1e-6)
        assert scaled.max() == pytest.approx(0.92138773)


    def test_fit_time_summary_per_feature(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        features = features.mean(1)
        scaler = MinMaxScaler()
        scaler.fit(features, (0,))

        expected_min = features.min(0)
        expected_max = features.max(0)
        np.testing.assert_array_almost_equal(scaler.min, expected_min)
        np.testing.assert_array_almost_equal(scaler.max, expected_max)


    def test_transform_time_summary_per_feature(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        features = features.mean(1)
        scaler = MinMaxScaler()
        scaler.fit(features, 0)

        scaled = scaler.transform(features)
        assert scaled.shape == features.shape
        assert scaled.min() == pytest.approx(0.0, abs=1e-6)
        assert scaled.max() == pytest.approx(1.0)


    def test_transform_time_summary_per_feature_slice(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        features = features.mean(1)
        scaler = MinMaxScaler()
        scaler.fit(features, 0)

        expected_min = features.min(0)
        expected_max = features.max(0)
        expected_scaled = (features[0] - expected_min) / (expected_max - expected_min)

        scaled = scaler.transform(features[0])
        assert scaled.shape == features[0].shape
        np.testing.assert_array_almost_equal(scaled, expected_scaled)
