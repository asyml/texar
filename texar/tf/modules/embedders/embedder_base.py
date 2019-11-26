# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The base embedder class.
"""

import tensorflow as tf

from texar.tf.core import layers
from texar.tf.module_base import ModuleBase


__all__ = [
    "EmbedderBase"
]


class EmbedderBase(ModuleBase):
    r"""The base embedder class that all embedder classes inherit.

    Args:
        num_embeds (int, optional): The number of embedding elements, e.g.,
            the vocabulary size of a word embedder.
        init_value (optional): A Tensor or numpy array that contains the
            initial value of embeddings. It is typically of shape
            ``[vocab_size] + embedding-dim``. Embedding can have dimensionality
            > 1.

            If `None`, embedding is initialized as specified in
            ``hparams["initializer"]``. Otherwise, the
            ``"initializer"`` and ``"dim"`` hyperparameters in
            :attr:`hparams` are ignored.
        hparams (dict or HParams, optional): Embedder hyperparameters. Missing
            hyperparamerters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure and
            default values.
    """

    def __init__(self, num_embeds=None, init_value=None, hparams=None):
        super().__init__(hparams=hparams)

        self._num_embeds = num_embeds
        self._init_value = init_value

    # pylint: disable=attribute-defined-outside-init
    def build(self, input_shape):
        r"""Build embedding layer.
        """
        with tf.name_scope('Embedding'):
            regularizer = layers.get_regularizer(self.hparams["regularizer"])
            if self._init_value is None:
                initializer = layers.get_initializer(
                    getattr(self.hparams, "initializer", None))
                dim = self.hparams["dim"]
                if not isinstance(self.hparams["dim"], (list, tuple)):
                    dim = [dim]
                if not initializer:
                    initializer = tf.initializers.GlorotUniform()
                self._embedding = self.add_weight(
                    name='w',
                    shape=[self._num_embeds] + dim,
                    initializer=initializer,
                    regularizer=regularizer,
                    trainable=self.hparams["trainable"]
                )
            else:
                init_value = tf.cast(self._init_value, tf.float32)
                self._embedding = tf.Variable(
                    name='w',
                    initial_value=init_value,
                    trainable=self.hparams["trainable"])

            self._num_embeds = self._embedding.shape[0]

            self._dim = self._embedding.shape[1:].as_list()
            self._dim_rank = len(self._dim)
            if self._dim_rank == 1:
                self._dim = self._dim[0]

        super().build(input_shape)

    def _get_dropout_layer(self, hparams, ids_rank=None, dropout_input=None,
                           dropout_strategy=None):
        r"""Creates dropout layer according to dropout strategy.
        Called in :meth:`call`.
        """
        dropout_layer = None

        st = dropout_strategy
        st = hparams.dropout_strategy if st is None else st

        if hparams.dropout_rate > 0.:
            if st == 'element':
                noise_shape = None
            elif st == 'item':
                assert dropout_input is not None
                assert ids_rank is not None
                noise_shape = (dropout_input.shape[:ids_rank]
                               + [1] * self._dim_rank)
            elif st == 'item_type':
                noise_shape = [None] + [1] * self._dim_rank
            else:
                raise ValueError('Unknown dropout strategy: {}'.format(st))

            dropout_layer = tf.keras.layers.Dropout(
                rate=hparams.dropout_rate, noise_shape=noise_shape)

        return dropout_layer

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python
            {
                "name": "embedder"
            }
        """
        return {
            "name": "embedder"
        }

    @property
    def num_embeds(self):
        r"""The number of embedding elements.
        """
        return self._num_embeds
