"""
Tests for StandardScaler class
"""

import pytest
import numpy as np

from spiegelib.features import StandardScaler

class TestStandardScaler():

    def test_empty_construction(self):
        scaler = StandardScaler()


    def test_fitting_full(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        scaler = StandardScaler()
        scaler.fit(features)
        assert scaler.mean == pytest.approx(-12.315056)
        assert scaler.std == pytest.approx(151.7704)

    def test_fit_and_transform_full(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        scaler = StandardScaler()

        scaler.fit(features)
        assert scaler.mean == pytest.approx(-12.315056)
        assert scaler.std == pytest.approx(151.7704)

        scaled = scaler.transform(features)
        assert scaled.mean() == pytest.approx(0.0, abs=1e-9)
        assert scaled.std() == pytest.approx(1.0)

    def test_fit_transform_full(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        scaler = StandardScaler()

        scaled = scaler.fit_transform(features)
        assert scaler.mean == pytest.approx(-12.315056)
        assert scaler.std == pytest.approx(151.7704)
        assert scaled.mean() == pytest.approx(0.0, abs=1e-9)
        assert scaled.std() == pytest.approx(1.0)

    def test_transform_slice(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        scaler = StandardScaler()
        scaler.fit(features)

        scaled = scaler.transform(features[0])
        assert scaled.mean() == pytest.approx(-0.036157943)
        assert scaled.std() == pytest.approx(0.8281236)


    def test_fit_per_feature(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        scaler = StandardScaler()
        scaler.fit(features, (0,1))

        expected_mean = np.load((shared_datadir / 'test_mfcc/per_feature_mean.npy').resolve())
        np.testing.assert_array_almost_equal(scaler.mean, expected_mean)

        expected_std = np.load((shared_datadir / 'test_mfcc/per_feature_std.npy').resolve())
        np.testing.assert_array_almost_equal(scaler.std, expected_std)


    def test_fit_and_transform_per_feature(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        scaler = StandardScaler()
        scaler.fit(features, (0,1))
        scaled = scaler.transform(features)

        expected_mean = np.zeros((features.shape[2],))
        np.testing.assert_array_almost_equal(scaled.mean((0,1)), expected_mean, 1e-5)

        expected_std = np.full(features.shape[2], 1)
        np.testing.assert_array_almost_equal(scaled.std((0,1)), expected_std, 1e-5)


    def test_transform_per_feature_slice(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        scaler = StandardScaler()
        scaler.fit(features, (0,1))
        scaled = scaler.transform(features[0])

        assert scaled.shape == features[0].shape
        assert scaled.mean() == pytest.approx(-0.2528363)
        assert scaled.std() == pytest.approx(0.8721026)


    def test_fit_and_transform_per_feature_non_time_major(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        features = np.swapaxes(features, 1, 2)[0:13]
        scaler = StandardScaler()
        scaler.fit(features, (0,2))
        scaled = scaler.transform(features)

        expected_mean = np.zeros((features.shape[1],))
        np.testing.assert_array_almost_equal(scaled.mean((0,2)), expected_mean)

        expected_std = np.full(features.shape[1], 1)
        np.testing.assert_array_almost_equal(scaled.std((0,2)), expected_std)


    def test_transform_per_feature_slice_non_time_major(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        features = np.swapaxes(features, 1, 2)
        scaler = StandardScaler()
        scaler.fit(features, (0,2))
        scaled = scaler.transform(features[0])

        assert scaled.shape == features[0].shape
        assert scaled.mean() == pytest.approx(-0.2528363)
        assert scaled.std() == pytest.approx(0.8721026)


    def test_fit_time_summary_full(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        features = features.mean(1)
        scaler = StandardScaler()
        scaler.fit(features)

        expected_mean = features.mean()
        expected_std = features.std()
        assert scaler.mean == pytest.approx(expected_mean)
        assert scaler.std == pytest.approx(expected_std)


    def test_transform_time_summary_full(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        features = features.mean(1)
        scaler = StandardScaler()
        scaler.fit(features)

        scaled = scaler.transform(features)
        assert scaled.shape == features.shape
        assert scaled.mean() == pytest.approx(0.0, abs=1e-6)
        assert scaled.std() == pytest.approx(1.0)


    def test_transform_time_summary_full_slice(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        features = features.mean(1)
        scaler = StandardScaler()
        scaler.fit(features)

        scaled = scaler.transform(features[0])
        assert scaled.shape == features[0].shape
        assert scaled.mean() == pytest.approx(-0.03666329, abs=1e-6)
        assert scaled.std() == pytest.approx(0.81805766)


    def test_fit_time_summary_per_feature(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        features = features.mean(1)
        scaler = StandardScaler()
        scaler.fit(features, (0,))

        expected_mean = features.mean(0)
        expected_std = features.std(0)
        np.testing.assert_array_almost_equal(scaler.mean, expected_mean)
        np.testing.assert_array_almost_equal(scaler.std, expected_std)


    def test_transform_time_summary_per_feature(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        features = features.mean(1)
        scaler = StandardScaler()
        scaler.fit(features, 0)

        scaled = scaler.transform(features)
        assert scaled.shape == features.shape
        assert scaled.mean() == pytest.approx(0.0, abs=1e-6)
        assert scaled.std() == pytest.approx(1.0)


    def test_transform_time_summary_per_feature_slice(self, shared_datadir):
        test_data = (shared_datadir / 'test_mfcc/train_features.npy').resolve()
        features = np.load(test_data)
        features = features.mean(1)
        scaler = StandardScaler()
        scaler.fit(features, 0)

        expected_mean = features.mean(0)
        expected_std = features.std(0)
        expected_scaled = (features[0] - expected_mean) / expected_std

        scaled = scaler.transform(features[0])
        assert scaled.shape == features[0].shape
        np.testing.assert_array_almost_equal(scaled, expected_scaled)
