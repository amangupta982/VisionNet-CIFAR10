"""
tests/test_predict.py
Unit tests for the prediction pipeline — output format, confidence, edge cases.
"""

import numpy as np
import pytest
import os
import tempfile
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.datasets import cifar10


CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def build_cnn(input_shape=(32, 32, 3), num_classes=10):
    model = Sequential([
        Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3,3), padding='same', activation='relu'),
        MaxPooling2D(2,2), Dropout(0.25),

        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        MaxPooling2D(2,2), Dropout(0.25),

        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3,3), padding='same', activation='relu'),
        MaxPooling2D(2,2), Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(), Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def predict_single(model, img_array):
    inp   = np.expand_dims(img_array, axis=0)
    probs = model.predict(inp, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return idx, probs[idx], probs


@pytest.fixture(scope="module")
def model():
    return build_cnn()


@pytest.fixture(scope="module")
def test_images():
    (_, _), (X_test, y_test) = cifar10.load_data()
    X_test = X_test[:50].astype('float32') / 255.0
    y_test = y_test[:50].flatten()
    return X_test, y_test


class TestPredictSingle:
    def test_returns_valid_class_index(self, model):
        img = np.random.rand(32, 32, 3).astype('float32')
        idx, conf, probs = predict_single(model, img)
        assert 0 <= idx <= 9

    def test_confidence_in_range(self, model):
        img = np.random.rand(32, 32, 3).astype('float32')
        _, conf, _ = predict_single(model, img)
        assert 0.0 <= conf <= 1.0

    def test_probs_length(self, model):
        img = np.random.rand(32, 32, 3).astype('float32')
        _, _, probs = predict_single(model, img)
        assert len(probs) == 10

    def test_probs_sum_to_one(self, model):
        img = np.random.rand(32, 32, 3).astype('float32')
        _, _, probs = predict_single(model, img)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_predicted_index_matches_argmax(self, model):
        img = np.random.rand(32, 32, 3).astype('float32')
        idx, conf, probs = predict_single(model, img)
        assert idx == int(np.argmax(probs))
        assert abs(conf - probs[idx]) < 1e-6

    def test_all_probs_non_negative(self, model):
        img = np.random.rand(32, 32, 3).astype('float32')
        _, _, probs = predict_single(model, img)
        assert np.all(probs >= 0.0)


class TestBatchPredictions:
    def test_batch_output_shape(self, model):
        batch = np.random.rand(10, 32, 32, 3).astype('float32')
        preds = model.predict(batch, verbose=0)
        assert preds.shape == (10, 10)

    def test_all_batch_probs_sum_to_one(self, model):
        batch = np.random.rand(20, 32, 32, 3).astype('float32')
        preds = model.predict(batch, verbose=0)
        sums  = preds.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_argmax_valid_for_all(self, model):
        batch   = np.random.rand(20, 32, 32, 3).astype('float32')
        preds   = model.predict(batch, verbose=0)
        indices = np.argmax(preds, axis=1)
        assert np.all((indices >= 0) & (indices <= 9))


class TestExternalImagePreprocessing:
    def test_jpeg_preprocessing(self):
        """Create a temp JPEG and verify it preprocesses to correct shape."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            tmp_path = f.name
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(tmp_path)

        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        loaded = load_img(tmp_path, target_size=(32, 32))
        arr    = img_to_array(loaded).astype('float32') / 255.0

        assert arr.shape  == (32, 32, 3)
        assert arr.min()  >= 0.0
        assert arr.max()  <= 1.0
        os.unlink(tmp_path)

    def test_png_preprocessing(self):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            tmp_path = f.name
        img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        img.save(tmp_path)

        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        loaded = load_img(tmp_path, target_size=(32, 32))
        arr    = img_to_array(loaded).astype('float32') / 255.0

        assert arr.shape == (32, 32, 3)
        os.unlink(tmp_path)

    def test_large_image_resizes_correctly(self):
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            tmp_path = f.name
        img = Image.fromarray(np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8))
        img.save(tmp_path)

        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        loaded = load_img(tmp_path, target_size=(32, 32))
        arr    = img_to_array(loaded).astype('float32') / 255.0

        assert arr.shape == (32, 32, 3)
        os.unlink(tmp_path)


class TestClassNames:
    def test_exactly_ten_classes(self):
        assert len(CLASS_NAMES) == 10

    def test_all_class_names_are_strings(self):
        assert all(isinstance(c, str) for c in CLASS_NAMES)

    def test_no_duplicate_class_names(self):
        assert len(set(CLASS_NAMES)) == len(CLASS_NAMES)

    def test_expected_classes_present(self):
        expected = {'airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck'}
        assert set(CLASS_NAMES) == expected
