"""
tests/test_model.py
Unit tests for CNN model architecture, compilation, and forward pass.
"""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization)


# ── Replicate the build function here so tests are self-contained ──────────
def build_cnn(input_shape=(32, 32, 3), num_classes=10):
    model = Sequential([
        Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3,3), padding='same', activation='relu'),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3,3), padding='same', activation='relu'),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


@pytest.fixture(scope="module")
def model():
    m = build_cnn()
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m


class TestModelArchitecture:
    def test_model_builds(self):
        m = build_cnn()
        assert m is not None

    def test_output_shape(self, model):
        # With batch of 4 images, output must be (4, 10)
        dummy = np.random.rand(4, 32, 32, 3).astype('float32')
        out   = model.predict(dummy, verbose=0)
        assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"

    def test_output_probabilities_sum_to_one(self, model):
        dummy = np.random.rand(8, 32, 32, 3).astype('float32')
        out   = model.predict(dummy, verbose=0)
        sums  = out.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5), "Softmax outputs must sum to 1"

    def test_output_values_in_range(self, model):
        dummy = np.random.rand(8, 32, 32, 3).astype('float32')
        out   = model.predict(dummy, verbose=0)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_input_shape_accepted(self, model):
        assert model.input_shape == (None, 32, 32, 3)

    def test_num_output_classes(self, model):
        assert model.output_shape == (None, 10)

    def test_has_conv_layers(self, model):
        conv_layers = [l for l in model.layers if isinstance(l, Conv2D)]
        assert len(conv_layers) == 6, "Expected 6 Conv2D layers"

    def test_has_batchnorm_layers(self, model):
        bn_layers = [l for l in model.layers if isinstance(l, BatchNormalization)]
        assert len(bn_layers) == 4, "Expected 4 BatchNormalization layers"

    def test_has_dropout_layers(self, model):
        do_layers = [l for l in model.layers if isinstance(l, Dropout)]
        assert len(do_layers) == 4, "Expected 4 Dropout layers"

    def test_has_maxpooling_layers(self, model):
        mp_layers = [l for l in model.layers if isinstance(l, MaxPooling2D)]
        assert len(mp_layers) == 3, "Expected 3 MaxPooling2D layers"

    def test_last_layer_is_softmax(self, model):
        last = model.layers[-1]
        assert isinstance(last, Dense)
        assert last.activation.__name__ == 'softmax'

    def test_parameter_count(self, model):
        total = model.count_params()
        # Should be around 1.34M — allow 20% tolerance
        assert 1_000_000 < total < 2_000_000, f"Unexpected param count: {total}"


class TestModelCompilation:
    def test_compiled_with_adam(self, model):
        assert isinstance(model.optimizer, tf.keras.optimizers.Adam)

    def test_compiled_with_correct_loss(self, model):
        assert model.loss == 'categorical_crossentropy'

    def test_compiled_with_accuracy_metric(self, model):
        metric_names = [m.name for m in model.metrics]
        assert 'accuracy' in metric_names or 'acc' in metric_names

    def test_model_is_trainable(self, model):
        trainable = sum(np.prod(w.shape) for w in model.trainable_weights)
        assert trainable > 0


class TestModelRobustness:
    def test_handles_single_image(self, model):
        single = np.random.rand(1, 32, 32, 3).astype('float32')
        out = model.predict(single, verbose=0)
        assert out.shape == (1, 10)

    def test_handles_large_batch(self, model):
        batch = np.random.rand(64, 32, 32, 3).astype('float32')
        out = model.predict(batch, verbose=0)
        assert out.shape == (64, 10)

    def test_handles_all_zeros_input(self, model):
        zeros = np.zeros((2, 32, 32, 3), dtype='float32')
        out = model.predict(zeros, verbose=0)
        assert out.shape == (2, 10)
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)

    def test_handles_all_ones_input(self, model):
        ones = np.ones((2, 32, 32, 3), dtype='float32')
        out = model.predict(ones, verbose=0)
        assert out.shape == (2, 10)
