# Copyright 2018 The Texar Authors. All Rights Reserved.
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

    Args:
        num_embeds (int, optional): The number of embedding elements, e.g.,
            the vocabulary size of a word embedder.
        hparams (dict or HParams, optional): Embedder hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.
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

        .. code-block:: python

            {
                "name": "embedder"
            }
        """
        return {
            "name": "embedder"
        }

    def _build(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def num_embeds(self):
        """The number of embedding elements.
        """
        return self._num_embeds
