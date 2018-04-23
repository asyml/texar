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

class FeedForwardNetwork(FeedForwardNetworkBase):
    """Feed forward neural network that consists of a sequence of layers.

    Args:
        layers (list, optional): A list of :tf_main:`Layer <layers/Layer>`
            instances composing the network. If given,
            :attr:`hparams['layers']` will be ignored.
        hparams (optional): A `dict` or an :class:`~texar.HParams` instance
            containing the hyperparameters. See :meth:`default_hparams` for
            valid hyperparameters.
    """

    def __init__(self, layers=None, hparams=None):
        FeedForwardNetworkBase.__init__(self, hparams)

        with tf.variable_scope(self.variable_scope):
            _build_layers(
                self, layers=layers, layer_hparams=self._hparams.layers)

    #TODO(zhiting): complete the docs
    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        TODO
        """
        return {
            "layers": [],
            "name": "NN"
        }

