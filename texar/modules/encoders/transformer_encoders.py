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
Transformer encoders with multihead self attention.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.core import layers
from texar.utils import transformer_attentions as attn
from texar.modules.encoders.encoder_base import EncoderBase
from texar.modules.encoders.multihead_attention import MultiheadAttentionEncoder
from texar.modules.networks.networks import FeedForwardNetwork
from texar import utils
from texar.utils.shapes import shape_list
from texar.utils.mode import is_train_mode

# pylint: disable=too-many-locals, invalid-name
# pylint: disable=arguments-differ, too-many-branches, too-many-statements

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
                        "units": output_dim*4,
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
                    "units": output_dim*4,
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

    This module basically stacks
    :class:`~texar.modules.encoders.MultiheadAttentionEncoder`,
    :class:`~texar.modules.FeedForwardNetwork` and residual connections.

    This module supports two types of architectures, namely, the standard
    Transformer Encoder architecture first proposed in
    `(Vaswani et al.) "Attention is All You Need"`, and
    the variant first used in `(Devlin et al.)` BERT. See
    :meth:`default_hparams` for the nuance between the two types of
    architectures.

    Args:
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
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

            self.multihead_attention_list = []
            self.poswise_networks = []
            for i in range(self._hparams.num_blocks):
                with tf.variable_scope("layer_{}".format(i)):

                    with tf.variable_scope('attention'):
                        mh_attn = MultiheadAttentionEncoder(
                            self._hparams.multihead_attention)
                        self.multihead_attention_list.append(mh_attn)

                        if self._hparams.dim != mh_attn.hparams.output_dim:
                            raise ValueError(
                                'The "dim" in the hparams of '
                                '"multihead_attention" should be equal to the '
                                '"dim" of TransformerEncoder')

                    pw_net = FeedForwardNetwork(
                        hparams=self._hparams['poswise_feedforward'])
                    final_dim = pw_net.hparams.layers[-1]['kwargs']['units']
                    if self._hparams.dim != final_dim:
                        raise ValueError(
                            'The output dimenstion of '
                            '"poswise_feedforward" should be equal '
                            'to the "dim" of TransformerEncoder.')
                    self.poswise_networks.append(pw_net)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "num_blocks": 6,
                "dim": 512,
                'use_bert_config': False,
                "embedding_dropout": 0.1,
                "residual_dropout": 0.1,
                "poswise_feedforward": default_transformer_poswise_net_hparams,
                'multihead_attention': {
                    'name': 'multihead_attention',
                    'num_units': 512,
                    'output_dim': 512,
                    'num_heads': 8,
                    'dropout_rate': 0.1,
                    'output_dim': 512,
                    'use_bias': False,
                },
                "initializer": None,
                "name": "transformer_encoder"
            }

        Here:

        "num_blocks" : int
            Number of stacked blocks.

        "dim" : int
            Hidden dimension of the encoders.

        "use_bert_config" : bool
            If `False`, apply the standard Transformer Encoder architecture from
            the original paper `(Vaswani et al.) "Attention is All You Need"`.
            If `True`, apply the Transformer Encoder architecture used in BERT
            `(Devlin et al.)`.

            The differences lie in:

                1. The standard arch restricts the word embedding of PAD token \
                   to all zero. The BERT arch does not.

                2. The attention bias for padding tokens: \
                   The standard arch uses `-1e8` for nagative attention mask. \
                   BERT uses `-1e4` instead.

                3. The residual connections between internal tensors: \
                   In BERT, a residual layer connects the tensors *after* \
                   layer normalization. In the standard arch, the tensors are \
                   connected *before* layer normalization.

        "embedding_dropout" : float
            Dropout rate of the input embedding.

        "residual_dropout" :  float
            Dropout rate of the residual connections.

        "poswise_feedforward" : dict
            Hyperparameters for a feed-forward network used in residual
            connections.
            Make sure the dimension of the output tensor is equal to `dim`.

            See :func:`~texar.modules.default_transformer_poswise_net_hparams`
            for details.

        "multihead_attention" : dict
            Hyperparameters for the multihead attention strategy.
            Make sure the "output_dim" in this module is equal to "dim".
            See :func:`~texar.modules.MultiheadAttentionEncoder.default_harams`
            for details.

        "initializer" : dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.
            See :func:`~texar.core.get_initializer` for details.

        "name" : str
            Name of the module.
        """
        return {
            'num_blocks': 6,
            'dim': 512,
            'use_bert_config': False,
            'embedding_dropout': 0.1,
            'residual_dropout': 0.1,
            'poswise_feedforward': default_transformer_poswise_net_hparams(),
            'multihead_attention': {
                'name': 'multihead_attention',
                'num_units': 512,
                'num_heads': 8,
                'dropout_rate': 0.1,
                'output_dim': 512,
                'use_bias': False,
            },
            'initializer': None,
            'name': 'transformer_encoder',
        }

    def _build(self, inputs, sequence_length, mode=None):
        """Encodes the inputs.

        Args:
            inputs: A 3D Tensor of shape `[batch_size, max_time, dim]`,
                containing the embedding of input sequences. Note that
                the embedding dimension `dim` must equal "dim" in
                :attr:`hparams`. The input embedding is typically an aggregation
                of word embedding and position embedding.
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

        inputs_padding = 1 - tf.sequence_mask(
            sequence_length, tf.shape(inputs)[1], dtype=tf.float32)
        if self._hparams.use_bert_config:
            ignore_padding = attn.attention_bias_ignore_padding(
                inputs_padding, bias_value=-1e4)
        else:
            ignore_padding = attn.attention_bias_ignore_padding(
                inputs_padding)
        encoder_self_attention_bias = ignore_padding

        input_embedding = inputs

        if self._hparams.use_bert_config:
            x = layers.layer_normalize(input_embedding)
            x = tf.layers.dropout(x,
                                  rate=self._hparams.embedding_dropout,
                                  training=is_train_mode(mode))
        else:
            x = tf.layers.dropout(input_embedding,
                                  rate=self._hparams.embedding_dropout,
                                  training=is_train_mode(mode))

        # Just to keep consistent with BERT, actually makes no difference
        if self._hparams.use_bert_config:
            pad_remover = None
        else:
            pad_remover = utils.transformer_utils.PadRemover(inputs_padding)

        for i in range(self._hparams.num_blocks):
            with tf.variable_scope("layer_{}".format(i)):
                multihead_attention = self.multihead_attention_list[i]

                # trivial difference between BERT and original Transformer
                if self._hparams.use_bert_config:
                    _queries_input = x
                else:
                    _queries_input = layers.layer_normalize(x)

                attention_output = multihead_attention(
                    queries=_queries_input,
                    memory=_queries_input,
                    memory_attention_bias=encoder_self_attention_bias,
                    mode=mode,
                )
                attention_output = tf.layers.dropout(
                    attention_output,
                    rate=self._hparams.residual_dropout,
                    training=is_train_mode(mode),
                )
                x = x + attention_output
                with tf.variable_scope('output'):
                    if self._hparams.use_bert_config:
                        x = layers.layer_normalize(x)
                        y = x
                    else:
                        y = layers.layer_normalize(x)

                poswise_network = self.poswise_networks[i]
                with tf.variable_scope(poswise_network.variable_scope):
                    original_shape = shape_list(y)
                    y = tf.reshape(y, [-1, self._hparams.dim])
                    if pad_remover:
                        y = tf.expand_dims(pad_remover.remove(y), axis=0)
                        # [1, batch_size*seq_length, hidden_dim]
                    layer_output = poswise_network(y, mode=mode)
                    sub_output = tf.layers.dropout(
                        layer_output,
                        rate=self._hparams.residual_dropout,
                        training=is_train_mode(mode)
                    )
                    if pad_remover:
                        sub_output = tf.reshape(pad_remover.restore(tf.squeeze(\
                            sub_output, axis=0)), original_shape \
                        )
                    else:
                        sub_output = tf.reshape(sub_output, original_shape)

                    x = x + sub_output
                    if self._hparams.use_bert_config:
                        x = layers.layer_normalize(x)

        if not self._hparams.use_bert_config:
            x = layers.layer_normalize(x)

        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

        return x
