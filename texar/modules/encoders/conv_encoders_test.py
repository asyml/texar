#
"""
Unit tests for conv encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

import texar as tx
from texar.modules.encoders.conv_encoders import Conv1DEncoder


class Conv1DEncoderTest(tf.test.TestCase):
    """Tests :class:`~texar.modules.Conv1DEncoder` class.
    """

    def test_encode(self):
        """Tests encode.
        """
        encoder_1 = Conv1DEncoder()
        self.assertEqual(len(encoder_1.layers), 4)
        self.assertTrue(isinstance(encoder_1.layer_by_name("conv_pool_1"),
                                   tx.core.MergeLayer))
        for layer in encoder_1.layers[0].layers:
            self.assertTrue(isinstance(layer, tx.core.SequentialLayer))

        inputs_1 = tf.ones([64, 16, 300], tf.float32)
        outputs_1 = encoder_1(inputs_1)
        self.assertEqual(outputs_1.shape, [64, 128])

        hparams = {
            # Conv layers
            "num_conv_layers": 2,
            "filters": 128,
            "kernel_size": [[3, 4, 5], 4],
            "other_conv_kwargs": {"padding": "same"},
            # Pooling layers
            "pooling": "AveragePooling",
            "pool_size": 2,
            "pool_strides": 1,
            # Dense layers
            "num_dense_layers": 3,
            "dense_size": [128, 128, 10],
            "dense_activation": "relu",
            "other_dense_kwargs": {"use_bias": False},
            # Dropout
            "dropout_conv": [0, 1, 2],
            "dropout_dense": 2
        }
        encoder_2 = Conv1DEncoder(hparams)
        # nlayers = nconv-pool + nconv + npool + ndense + ndropout + flatten
        self.assertEqual(len(encoder_2.layers), 1+1+1+3+4+1)
        self.assertTrue(isinstance(encoder_2.layer_by_name("conv_pool_1"),
                                   tx.core.MergeLayer))
        for layer in encoder_2.layers[1].layers:
            self.assertTrue(isinstance(layer, tx.core.SequentialLayer))

        inputs_2 = tf.ones([64, 16, 300], tf.float32)
        outputs_2 = encoder_2(inputs_2)
        self.assertEqual(outputs_2.shape, [64, 10])

    def test_unknown_seq_length(self):
        """Tests use of pooling layer when the seq_length dimension of inputs
        is `None`.
        """
        encoder_1 = Conv1DEncoder()
        inputs_1 = tf.placeholder(tf.float32, [64, None, 300])
        outputs_1 = encoder_1(inputs_1)
        self.assertEqual(outputs_1.shape, [64, 128])

        hparams = {
            # Conv layers
            "num_conv_layers": 2,
            "filters": 128,
            "kernel_size": [[3, 4, 5], 4],
            # Pooling layers
            "pooling": "AveragePooling",
            "pool_size": [2, None],
            # Dense layers
            "num_dense_layers": 1,
            "dense_size": 10,
        }
        encoder = Conv1DEncoder(hparams)
        # nlayers = nconv-pool + nconv + npool + ndense + ndropout + flatten
        self.assertEqual(len(encoder.layers), 1+1+1+1+1+1)
        self.assertTrue(isinstance(encoder.layer_by_name('pool_2'),
                                   tx.core.AverageReducePooling1D))

        inputs = tf.placeholder(tf.float32, [64, None, 300])
        outputs = encoder(inputs)
        self.assertEqual(outputs.shape, [64, 10])

        hparams_2 = {
            # Conv layers
            "num_conv_layers": 1,
            "filters": 128,
            "kernel_size": 4,
            "other_conv_kwargs": {'data_format': 'channels_first'},
            # Pooling layers
            "pooling": "MaxPooling",
            "other_pool_kwargs": {'data_format': 'channels_first'},
            # Dense layers
            "num_dense_layers": 1,
            "dense_size": 10,
        }
        encoder_2 = Conv1DEncoder(hparams_2)
        inputs_2 = tf.placeholder(tf.float32, [64, 300, None])
        outputs_2 = encoder_2(inputs_2)
        self.assertEqual(outputs_2.shape, [64, 10])


if __name__ == "__main__":
    tf.test.main()
