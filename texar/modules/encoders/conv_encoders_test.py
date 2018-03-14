#
"""
Unit tests for conv encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from texar.modules.encoders.conv_encoders import SimpleConv1DEncoder


class SimpleConv1DEncoderTest(tf.test.TestCase):
    """Tests :class:`~texar.modules.SimpleConv1DEncoder` class.
    """

    def test_network_structure(self):
        """Tests nn structure.
        """
        encoder_1 = SimpleConv1DEncoder()
        print(encoder_1.layer_names)

        #hparams = {
        #    # Conv layers
        #    "num_conv_layers": 1,
        #    "filters": 128,
        #    "kernel_size": [3, 4, 5],
        #    "conv_activation": "identity",
        #    "other_conv_kwargs": None,
        #    # Pooling layers
        #    "pooling": "MaxPooling1D",
        #    "pool_size": 1,
        #    "pool_strides": 1,
        #    "other_pool_kwargs": None,
        #    # Dense layers
        #    "num_dense_layers": 1,
        #    "dense_size": 128,
        #    "dense_activation": "identity",
        #    "other_dense_kwargs": None,
        #    # Dropout
        #    "dropout_conv": 1,
        #    "dropout_dense": [],
        #}

if __name__ == "__main__":
    tf.test.main()
