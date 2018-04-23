#
"""
Various neural networks and related utilities.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from texar.modules.networks.network_base import FeedForwardNetworkBase
from texar.modules.networks.network_base import _build_layers

__all__ = [
    "FeedForwardNetwork"
]

#TDDO(zhiting): complete the docs
class FeedForwardNetwork(FeedForwardNetworkBase):
    """Feed forward neural network that consists of a sequence of layers.

    Args:
        layers (list, optional):
    """

    def __init__(self, layers=None, hparams=None):
        FeedForwardNetworkBase.__init__(self, hparams)

        with tf.variable_scope(self.variable_scope):
            _build_layers(
                self, layers=layers, layer_hparams=self._hparams.layers)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        TODO
        """
        return {
            "layers": [],
            "name": "NN"
        }

