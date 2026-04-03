"""
tests/test_preprocessing.py
Unit tests for data loading and preprocessing pipeline.
"""

import numpy as np
import pytest
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator


@pytest.fixture(scope="module")
def cifar_data():
    """Load a small slice of CIFAR-10 once for all tests in this module."""
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # Use only first 500 samples to keep tests fast
    return X_train[:500], y_train[:500], X_test[:100], y_test[:100]


class TestDataLoading:
    def test_train_shape(self, cifar_data):
        X_train, y_train, _, _ = cifar_data
        assert X_train.shape[1:] == (32, 32, 3), "Images must be 32x32 RGB"

    def test_test_shape(self, cifar_data):
        _, _, X_test, y_test = cifar_data
        assert X_test.shape[1:] == (32, 32, 3)

    def test_label_range(self, cifar_data):
        _, y_train, _, y_test = cifar_data
        assert y_train.min() >= 0 and y_train.max() <= 9, "Labels must be 0-9"
        assert y_test.min()  >= 0 and y_test.max()  <= 9

    def test_pixel_dtype_before_norm(self, cifar_data):
        X_train, _, _, _ = cifar_data
        assert X_train.dtype == np.uint8, "Raw pixels must be uint8"

    def test_pixel_range_before_norm(self, cifar_data):
        X_train, _, _, _ = cifar_data
        assert X_train.min() >= 0 and X_train.max() <= 255


class TestNormalization:
    def test_pixel_range_after_norm(self, cifar_data):
        X_train, _, X_test, _ = cifar_data
        X_train_n = X_train.astype('float32') / 255.0
        X_test_n  = X_test.astype('float32')  / 255.0
        assert X_train_n.min() >= 0.0 and X_train_n.max() <= 1.0
        assert X_test_n.min()  >= 0.0 and X_test_n.max()  <= 1.0

    def test_normalization_dtype(self, cifar_data):
        X_train, _, _, _ = cifar_data
        X_norm = X_train.astype('float32') / 255.0
        assert X_norm.dtype == np.float32

    def test_normalization_preserves_shape(self, cifar_data):
        X_train, _, _, _ = cifar_data
        X_norm = X_train.astype('float32') / 255.0
        assert X_norm.shape == X_train.shape


class TestOneHotEncoding:
    def test_one_hot_shape(self, cifar_data):
        _, y_train, _, y_test = cifar_data
        y_cat = to_categorical(y_train, 10)
        assert y_cat.shape == (len(y_train), 10)

    def test_one_hot_values(self, cifar_data):
        _, y_train, _, _ = cifar_data
        y_cat = to_categorical(y_train, 10)
        # Each row must sum to 1 and contain exactly one '1'
        assert np.allclose(y_cat.sum(axis=1), 1.0)
        assert np.all((y_cat == 0) | (y_cat == 1))

    def test_one_hot_correct_class(self, cifar_data):
        _, y_train, _, _ = cifar_data
        y_cat = to_categorical(y_train, 10)
        # The argmax of each one-hot row should match the original label
        assert np.all(np.argmax(y_cat, axis=1) == y_train.flatten())

    def test_num_classes(self, cifar_data):
        _, y_train, _, _ = cifar_data
        assert len(np.unique(y_train)) == 10


class TestDataAugmentation:
    def test_augmentation_output_shape(self, cifar_data):
        X_train, y_train, _, _ = cifar_data
        X_norm  = X_train.astype('float32') / 255.0
        y_cat   = to_categorical(y_train, 10)
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        )
        datagen.fit(X_norm)
        batch_x, batch_y = next(datagen.flow(X_norm, y_cat, batch_size=32))
        assert batch_x.shape == (32, 32, 32, 3)
        assert batch_y.shape == (32, 10)

    def test_augmented_pixel_range(self, cifar_data):
        X_train, y_train, _, _ = cifar_data
        X_norm  = X_train.astype('float32') / 255.0
        y_cat   = to_categorical(y_train, 10)
        datagen = ImageDataGenerator(horizontal_flip=True)
        batch_x, _ = next(datagen.flow(X_norm, y_cat, batch_size=16))
        assert batch_x.min() >= 0.0 and batch_x.max() <= 1.0
