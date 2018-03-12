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

class SimpleConvEncoder(EncoderBase):
    """Simple conv encoder.
    """

    def __init__(self, hparams=None):
       pass

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        return {
            # Conv layers
            "conv_dim": 1,
            "num_conv_layers": 1,
            "filters": 128,
            "kernel_size": [3, 4, 5],
            "conv_activation": "identity",
            "other_conv_kwargs": {},
            # Pooling layers
            # Dense layers
            "num_dense_layers": 1,
            "dense_units": 128,
            "dense_activation": "identity",
            "other_dense_kwargs": {},
            # Dropout
            "dropout_conv_layers": 0,
            "dropout_dense_layers": 0,
            "name": "conv_encoder",
            "@no_typecheck": ["filters", "kernel_size"]
        }

