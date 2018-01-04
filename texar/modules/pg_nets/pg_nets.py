"""
Vanilla Policy Gradient Network
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.modules.pg_nets import PGNetBase
from texar.modules import FeedForwardNetwork


class PGNet(PGNetBase):
    """
    Vanilla Policy Gradient Network
    """
    def __init__(self, hparams=None):
        PGNetBase.__init__(self, hparams=hparams)
        with tf.variable_scope(self.variable_scope):
            self.network = FeedForwardNetwork(
                hparams=self.hparams.network_hparams)

    @staticmethod
    def default_hparams():
        return {
            'name': 'pg_net',
            'network_hparams': FeedForwardNetwork.default_hparams()
        }

    def _build(self, inputs):
        output = self.network(inputs)

        if not self._built:
            self._add_internal_trainable_variables()
            self._add_trainable_variable(self.network.trainable_variables)
            self._built = True

        return output
