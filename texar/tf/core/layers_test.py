"""
Unit tests for various layers.
"""

import numpy as np

import tensorflow as tf

from texar.tf.core import layers


class GetActivationFnTest(tf.test.TestCase):
    """Tests :func:`texar.tf.core.layers.get_activation_fn`.
    """
    def test_get_activation_fn(self):
        """Tests.
        """
        fn = layers.get_activation_fn()
        self.assertEqual(fn, tf.identity)

        fn = layers.get_activation_fn('relu')
        self.assertEqual(fn, tf.nn.relu)

        inputs = tf.random.uniform([64, 100], -5, 20, dtype=tf.int32)

        fn = layers.get_activation_fn('leaky_relu')
        fn_output = fn(inputs)
        ref_output = tf.nn.leaky_relu(inputs)
        np.testing.assert_array_equal(fn_output, ref_output)

        fn = layers.get_activation_fn('leaky_relu', kwargs={'alpha': 0.1})
        fn_output = fn(inputs)
        ref_output = tf.nn.leaky_relu(inputs, alpha=0.1)
        np.testing.assert_array_equal(fn_output, ref_output)


class GetLayerTest(tf.test.TestCase):
    """Tests layer creator.
    """
    def test_get_layer(self):
        """Tests :func:`texar.tf.core.layers.get_layer`.
        """
        hparams = {
            "type": "Conv1D"
        }
        layer = layers.get_layer(hparams)
        self.assertTrue(isinstance(layer, tf.keras.layers.Conv1D))

        hparams = {
            "type": tf.keras.layers.Conv1D
        }
        layer = layers.get_layer(hparams)
        self.assertTrue(isinstance(layer, tf.keras.layers.Conv1D))

        hparams = {
            "type": tf.keras.layers.Conv1D(filters=10, kernel_size=2)
        }
        layer = layers.get_layer(hparams)
        self.assertTrue(isinstance(layer, tf.keras.layers.Conv1D))


if __name__ == "__main__":
    tf.test.main()
