from q_network_base import QNetworkBase
from txtgen.modules.networks import FeedForwardNetwork

import tensorflow as tf


class NatureQNetwork(QNetworkBase):
    def __init__(self, hparams=None):
        QNetworkBase.__init__(self, hparams=hparams)
        with tf.variable_scope(self.variable_scope):
            self.qnet = FeedForwardNetwork(hparams=self.hparams.network_hparams)
            self.target = FeedForwardNetwork(hparams=self.hparams.network_hparams)

    @staticmethod
    def default_hparams():
        return {
            'name': 'nature_q_network',
            'network_hparams': FeedForwardNetwork.default_hparams()
        }

    def _build(self, inputs):
        qnet_result, target_result = self.qnet(inputs), self.target(inputs)
        if not self._built:
            self._add_internal_trainable_variables()
            self._add_trainable_variable(self.qnet.trainable_variables)
            self._add_trainable_variable(self.target.trainable_variables)
            self._built = True
        return qnet_result, target_result
