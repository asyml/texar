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
Transformer encoders with multihead self attention.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.core import layers
from texar.utils import transformer_attentions as attn
from texar.modules.embedders.position_embedders import SinusoidsPositionEmbedder
from texar.modules.encoders.encoder_base import EncoderBase
from texar.modules.networks.networks import FeedForwardNetwork
from texar import utils
from texar.utils.shapes import shape_list, mask_sequences
from texar.utils.mode import is_train_mode

# pylint: disable=too-many-locals, invalid-name

__all__ = [
    "default_transformer_poswise_net_hparams",
    "TransformerEncoder"
]

def default_transformer_poswise_net_hparams(output_dim=512):
    """Returns default hyperparameters of a
    :class:`~texar.modules.FeedForwardNetwork` as a pos-wise network used
    in :class:`~texar.modules.TransformerEncoder` and
    :class:`~texar.modules.TransformerDecoder`.

    This is a 2-layer dense network with dropout in-between.

    .. code-block:: python

        {
            "layers": [
                {
                    "type": "Dense",
                    "kwargs": {
                        "name": "conv1",
                        "units": 2048,
                        "activation": "relu",
                        "use_bias": True,
                    }
                },
                {
                    "type": "Dropout",
                    "kwargs": {
                        "rate": 0.1,
                    }
                },
                {
                    "type": "Dense",
                    "kwargs": {
                        "name": "conv2",
                        "units": output_dim,
                        "use_bias": True,
                    }
                }
            ],
            "name": "ffn"
        }

    Args:
        output_dim (int): The size of output dense layer.
    """
    return {
        "layers": [
            {
                "type": "Dense",
                "kwargs": {
                    "name": "conv1",
                    "units": 2048,
                    "activation": "relu",
                    "use_bias": True,
                }
            },
            {
                "type": "Dropout",
                "kwargs": {
                    "rate": 0.1,
                }
            },
            {
                "type": "Dense",
                "kwargs": {
                    "name": "conv2",
                    "units": output_dim,
                    "use_bias": True,
                }
            }
        ],
        "name": "ffn"
    }


class TransformerEncoder(EncoderBase):
    """Transformer encoder that applies multi-head self attention for encoding
    sequences.

    Args:
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    .. document private functions
    .. automethod:: _build
    """
    def __init__(self, hparams=None):
        EncoderBase.__init__(self, hparams)

        with tf.variable_scope(self.variable_scope):
            if self._hparams.initializer:
                tf.get_variable_scope().set_initializer(
                    layers.get_initializer(self._hparams.initializer))

            self.position_embedder = \
                SinusoidsPositionEmbedder(
                    self._hparams.position_embedder_hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "num_blocks": 6,
                "num_heads": 8,
                "dim": 512,
                "position_embedder_hparams": None,
                "embedding_dropout": 0.1,
                "attention_dropout": 0.1,
                "residual_dropout": 0.1,
                "poswise_feedforward": default_transformer_poswise_net_hparams,
                "initializer": None,
                "name": "transformer_encoder"
            }

        Here:

        "num_blocks" : int
            Number of stacked blocks.

        "num_heads" : int
            Number of heads for attention calculation.

        "dim" : int
            Hidden dimension of the encoder.

        "position_embedder_hparams" : dict, optional
            Hyperparameters of a
            :class:`~texar.modules.SinusoidsPositionEmbedder` as position
            embedder. If `None`, the
            :meth:`~texar.modules.SinusoidsPositionEmbedder.default_hparams`
            is used.

        "embedding_dropout" : float
            Dropout rate of the input word and position embeddings.

        "attention_dropout: : float
            Dropout rate in the attention.

        "residual_dropout" :  float
            Dropout rate of the residual connections.

        "poswise_feedforward" : dict,
            Hyperparameters for a feed-forward network used in residual
            connections.

            See :func:`~texar.modules.default_transformer_poswise_net_hparams`
            for details.

        "initializer" : dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.
            See :func:`~texar.core.get_initializer` for details.

        "name" : str
            Name of the module.
        """
        return {
            'initializer': None,
            'position_embedder_hparams': None,
            'embedding_dropout': 0.1,
            'attention_dropout': 0.1,
            'residual_dropout': 0.1,
            'num_blocks': 6,
            'num_heads': 8,
            'poswise_feedforward': default_transformer_poswise_net_hparams(),
            'dim': 512,
            'name': 'transformer_encoder',
        }

    # pylint: disable=arguments-differ
    def _build(self, inputs, sequence_length, mode=None):
        """Encodes the inputs.

        Args:
            inputs: A 3D Tensor of shape `[batch_size, max_time, dim]`,
                containing the word embeddings of input sequences. Note that
                the embedding dimension `dim` must equal "dim" in
                :attr:`hparams`.
            sequence_length: A 1D Tensor of shape `[batch_size]`. Input tokens
                beyond respective sequence lengths are masked out
                automatically.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`,
                including `TRAIN`, `EVAL`, and `PREDICT`. Used to toggle
                dropout.
                If `None` (default), :func:`texar.global_mode` is used.

        Returns:
            A Tensor of shape `[batch_size, max_time, dim]` containing the
            encoded vectors.
        """
        # Multiply input embedding with the sqrt of its dimension for
        # normalization
        inputs = inputs * self._hparams.dim**0.5

        inputs = mask_sequences(inputs, sequence_length, tensor_rank=3)

        _, lengths, _ = shape_list(inputs)

        inputs_padding = 1 - tf.sequence_mask(
            sequence_length, tf.shape(inputs)[1], dtype=tf.float32)
        ignore_padding = attn.attention_bias_ignore_padding(inputs_padding)
        encoder_self_attention_bias = ignore_padding

        pos_embeds = self.position_embedder(lengths,
                                            self._hparams.dim)
        input_embedding = inputs + pos_embeds

        x = tf.layers.dropout(input_embedding,
                              rate=self._hparams.embedding_dropout,
                              training=is_train_mode(mode))
        pad_remover = utils.transformer_utils.PadRemover(inputs_padding)

        for i in range(self._hparams.num_blocks):
            with tf.variable_scope("layer_{}".format(i)):
                with tf.variable_scope('self_attention'):
                    selfatt_output = attn.multihead_attention(
                        queries=layers.layer_normalize(x),
                        memory=None,
                        memory_attention_bias=encoder_self_attention_bias,
                        num_heads=self._hparams.num_heads,
                        dropout_rate=self._hparams.attention_dropout,
                        num_units=self._hparams.dim,
                        scope='multihead_attention'
                    )
                    x = x + tf.layers.dropout(
                        selfatt_output,
                        rate=self._hparams.residual_dropout,
                        training=is_train_mode(mode),
                    )

                poswise_network = FeedForwardNetwork(
                    hparams=self._hparams['poswise_feedforward'])
                with tf.variable_scope(poswise_network.variable_scope):
                    y = layers.layer_normalize(x)
                    original_shape = shape_list(y)
                    y = tf.reshape(y, [-1, self._hparams.dim])
                    y = tf.expand_dims(pad_remover.remove(y), axis=0)
                    # [1, batch_size*seq_length, hidden_dim]
                    sub_output = tf.layers.dropout(
                        poswise_network(y),
                        rate=self._hparams.residual_dropout,
                        training=is_train_mode(mode)
                    )
                    sub_output = tf.reshape(pad_remover.restore(tf.squeeze(\
                        sub_output, axis=0)), original_shape \
                    )
                    x = x + sub_output

        encoder_output = layers.layer_normalize(x)

        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

        return encoder_output
