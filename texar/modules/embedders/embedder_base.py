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

        self._dropout_layer = None
        if hparams.dropout_rate > 0.:
            with tf.variable_scope(tf.variable_scope):
                self._dropout_layer = tf.layers.Dropout(
                    rate=hparams.dropout_rate)

        self._num_embeds = self._embedding.get_shape().as_list()[0]

        self._dim = self._embedding.get_shape().as_list()[1:]
        if len(self._dim) == 1:
            self._dim = self._dim[0]


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
