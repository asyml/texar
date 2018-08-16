#
"""
The base embedder class.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.module_base import ModuleBase
from texar.modules.embedders import embedder_utils

# pylint: disable=invalid-name

__all__ = [
    "EmbedderBase"
]

class EmbedderBase(ModuleBase):
    """The base embedder class that all embedder classes inherit.
    """

    def __init__(self, num_embeds=None, hparams=None):
        ModuleBase.__init__(self, hparams)

        self._num_embeds = num_embeds

    # pylint: disable=attribute-defined-outside-init
    def _init_parameterized_embedding(self, init_value, num_embeds, hparams):
        self._embedding = embedder_utils.get_embedding(
            hparams, init_value, num_embeds, self.variable_scope)
        if hparams.trainable:
            self._add_trainable_variable(self._embedding)

        self._num_embeds = self._embedding.get_shape().as_list()[0]

        self._dim = self._embedding.get_shape().as_list()[1:]
        self._dim_rank = len(self._dim)
        if self._dim_rank == 1:
            self._dim = self._dim[0]

    def _get_dropout_layer(self, hparams, ids_rank=None, dropout_input=None,
                           dropout_strategy=None):
        """Creates dropout layer according to dropout strategy.

        Called in :meth:`_build()`.
        """
        dropout_layer = None

        st = dropout_strategy
        st = hparams.dropout_strategy if st is None else st

        if hparams.dropout_rate > 0.:
            if st == 'element':
                noise_shape = None
            elif st == 'item':
                noise_shape = tf.concat([tf.shape(dropout_input)[:ids_rank],
                                         tf.ones([self._dim_rank], tf.int32)],
                                        axis=0)
            elif st == 'item_type':
                noise_shape = [None] + [1] * self._dim_rank
            else:
                raise ValueError('Unknown dropout strategy: {}'.format(st))

            dropout_layer = tf.layers.Dropout(
                rate=hparams.dropout_rate, noise_shape=noise_shape)

        return dropout_layer

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        return {
            "name": "embedder"
        }

    def _build(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def num_embeds(self):
        """The number of embedding vectors.
        """
        return self._num_embeds
