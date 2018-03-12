#
"""
Various convolutional network encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.modules.encoders.encoder_base import EncoderBase
from texar.modules.networks import FeedForwardNetwork
from texar.core import layers

# pylint: disable=not-context-manager, too-many-arguments

__all__ = [
]

class SimpleConv1DEncoder(EncoderBase):
    """Simple Conv-1D encoder.
    """

    def __init__(self, hparams=None):
       pass

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        return {
            # Conv layers
            "num_conv_layers": 1,
            "filters": 128,
            "kernel_size": [3, 4, 5],
            "conv_activation": "identity",
            "other_conv_kwargs": {},
            # Pooling layers
            "pooling": "MaxPooling1D",
            "pool_size": 1,
            "pool_strides": 1,
            "other_pool_kwargs": {},
            # Dense layers
            "num_dense_layers": 1,
            "dense_size": 128,
            "dense_activation": "identity",
            "other_dense_kwargs": {},
            # Dropout
            "dropout_conv": 1,
            "dropout_dense": [],
            "dropout_rate": 0.75,
            # Others
            "name": "conv_encoder",
            "@no_typecheck": ["filters", "kernel_size",
                              "dropout_conv", "dropout_dense"]
        }

