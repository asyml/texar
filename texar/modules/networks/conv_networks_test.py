#
"""
Unit tests for conv networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

import texar as tx
from texar.modules.networks.conv_networks import Conv1DNetwork


class Conv1DNetworkTest(tf.test.TestCase):
    """Tests :class:`~texar.modules.Conv1DNetwork` class.
    """

    def test_feedforward(self):
        """Tests feed forward.
        """
        network_1 = Conv1DNetwork()
        self.assertEqual(len(network_1.layers), 4)
        self.assertTrue(isinstance(network_1.layer_by_name("conv_pool_1"),
                                   tx.core.MergeLayer))
        for layer in network_1.layers[0].layers:
            self.assertTrue(isinstance(layer, tx.core.SequentialLayer))

        inputs_1 = tf.ones([64, 16, 300], tf.float32)
        outputs_1 = network_1(inputs_1)
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
        network_2 = Conv1DNetwork(hparams)
        # nlayers = nconv-pool + nconv + npool + ndense + ndropout + flatten
        self.assertEqual(len(network_2.layers), 1+1+1+3+4+1)
        self.assertTrue(isinstance(network_2.layer_by_name("conv_pool_1"),
                                   tx.core.MergeLayer))
        for layer in network_2.layers[1].layers:
            self.assertTrue(isinstance(layer, tx.core.SequentialLayer))

        inputs_2 = tf.ones([64, 16, 300], tf.float32)
        outputs_2 = network_2(inputs_2)
        self.assertEqual(outputs_2.shape, [64, 10])

    def test_unknown_seq_length(self):
        """Tests use of pooling layer when the seq_length dimension of inputs
        is `None`.
        """
        network_1 = Conv1DNetwork()
        inputs_1 = tf.placeholder(tf.float32, [64, None, 300])
        outputs_1 = network_1(inputs_1)
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
        network = Conv1DNetwork(hparams)
        # nlayers = nconv-pool + nconv + npool + ndense + ndropout + flatten
        self.assertEqual(len(network.layers), 1+1+1+1+1+1)
        self.assertTrue(isinstance(network.layer_by_name('pool_2'),
                                   tx.core.AverageReducePooling1D))

        inputs = tf.placeholder(tf.float32, [64, None, 300])
        outputs = network(inputs)
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
        network_2 = Conv1DNetwork(hparams_2)
        inputs_2 = tf.placeholder(tf.float32, [64, 300, None])
        outputs_2 = network_2(inputs_2)
        self.assertEqual(outputs_2.shape, [64, 10])

    def test_mask_input(self):
        """Tests masked inputs.
        """
        network_1 = Conv1DNetwork()
        inputs_1 = tf.ones([3, 16, 300], tf.float32)
        seq_length = [10, 15, 1]
        outputs_1 = network_1(inputs_1, sequence_length=seq_length)
        self.assertEqual(outputs_1.shape, [3, 128])


if __name__ == "__main__":
    tf.test.main()
