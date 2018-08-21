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
from texar.utils import transformer_attentions as attentions
from texar.modules.embedders import position_embedders
from texar.modules.encoders.encoder_base import EncoderBase
from texar.modules.networks.networks import FeedForwardNetwork
from texar import utils
from texar.utils.shapes import shape_list
from texar.utils.mode import is_train_mode

# pylint: disable=too-many-locals, invalid-name

__all__ = [
    "TransformerEncoder"
]

class TransformerEncoder(EncoderBase):
    """Transformer encoder.

    Args:
        vocab_size (int, optional): The vocabulary size. Required if
            :attr:`embedding` is not provided.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.
    """
    def __init__(self,
                 hparams=None):
        EncoderBase.__init__(self, hparams)
        self.enc = None
        with tf.variable_scope(self.variable_scope):
            if self._hparams.initializer:
                tf.get_variable_scope().set_initializer(
                    layers.get_initializer(self._hparams.initializer))
            if self._hparams.position_embedder.name == 'sinusoids':
                self.position_embedder = \
                    position_embedders.SinusoidsPositionEmbedder(\
                    self._hparams.position_embedder.hparams)
        self.stack_output = None

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        .. code-block:: python
            {
                'initializer':None,
                'multiply_embedding_mode': 'sqrt_depth',
                "position_embedder": None,
                "name":"encoder",
                "zero_pad":False,
                "bos_pad":False,
                'embedding_dropout':0.1,
                'attention_dropout':0.1,
                'residual_dropout':0.1,
                'num_blocks':6,
                'num_heads':8,
                'poswise_feedforward':None,
                'num_units': 512,
                "name": "transformer_encoder"
            }

        Here:
            initializer: The default initializer to initialize the
                variables created in this module.
            multiply_embedding_mode: 'sqrt_depth' is a normalization method
                for the following attention calculation, multiplying each
                embedding to the sqrt of its dimension.
            position_embedder: The hyperparameters to define the position
                embedder.
            zero_pad: If it's set as True, we use all-zero embedding to
                represent the special <PAD> token.
            bos_pad: If it's set as True, we use all-zero embedding to
                represent the special <BOS> token.
            embeddin_dropout: The dropout rate of the word embeddings.
            attention_dropout: The dropout rate of the attention
                calculation.
            residual_dropout, The dropout rate of the residual connection.
            num_blocks: The number of stacked blocks.
            num_heads: The number of heads for attention calculation.
            poswise_feedforward: The hyperparameters to define the
                feed forward layers in each block.
            num_units: The dimension the embeddings and encoded vectors.
            name: A tensorflow-compatible name for this module.
        """
        return {
            'initializer': None,
            'multiply_embedding_mode': 'sqrt_depth',
            "position_embedder": None,
            "zero_pad": False,
            "bos_pad": False,
            'embedding_dropout': 0.1,
            'attention_dropout': 0.1,
            'residual_dropout': 0.1,
            'num_blocks': 6,
            'num_heads': 8,
            'poswise_feedforward': None,
            'num_units': 512,
            "name": "encoder",
        }

    # pylint: disable=arguments-differ
    def _build(self, enc, inputs_padding, mode=None):
        """Encodes the inputs.

        Args:
            enc: A 3D Tensor of shape `[batch_size, max_time, dim]
            inputs_padding: A 2D Tensor to indicate which positions
                in the enc tensor are paddings.
            mode(optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`

        Returns:
            encoder_output: the encoded vectors
            encoder_decoder_attention_bias: The masks to indicate which
                positions in the inputs are not padding.
        """
        bsize, lengths, channels = shape_list(enc)
        if self._hparams.zero_pad:
            if not self._hparams.bos_pad:
                enc = tf.concat(\
                    (tf.zeros(shape=[bsize, 1, channels]),
                        enc[:, 1:, :]), 1)
            else:
                enc = tf.concat(\
                    (tf.zeros(shape=[bsize, 2, channels]),
                        enc[:, 2:, :]), 1)
        if self._hparams.multiply_embedding_mode == 'sqrt_depth':
            self.enc = enc * channels**0.5
        else:
            self.enc = enc
        ignore_padding = attentions.attention_bias_ignore_padding(
            inputs_padding)
        encoder_self_attention_bias = ignore_padding
        encoder_decoder_attention_bias = ignore_padding

        pos_embeds = self.position_embedder(lengths, channels)
        input_embedding = self.enc + pos_embeds

        x = tf.layers.dropout(input_embedding,
                              rate=self._hparams.embedding_dropout,
                              training=is_train_mode(mode))
        pad_remover = utils.transformer_utils.PadRemover(inputs_padding)
        for i in range(self._hparams.num_blocks):
            with tf.variable_scope("layer_{}".format(i)):
                with tf.variable_scope('self_attention'):
                    selfatt_output = attentions.multihead_attention(
                        queries=layers.layer_normalize(x),
                        memory=None,
                        memory_attention_bias=encoder_self_attention_bias,
                        num_heads=self._hparams.num_heads,
                        dropout_rate=self._hparams.attention_dropout,
                        num_units=self._hparams.num_units,
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
                    y = tf.reshape(y, [-1, self._hparams.num_units])
                    y = tf.expand_dims(pad_remover.remove(y), axis=0)
                    #[1, batch_size*seq_length, hidden_dim]
                    sub_output = tf.layers.dropout(
                        poswise_network(y),
                        rate=self._hparams.residual_dropout,
                        training=is_train_mode(mode)
                    )
                    sub_output = tf.reshape(pad_remover.restore(tf.squeeze(\
                        sub_output, axis=0)), original_shape \
                    )
                    x = x + sub_output

        self.stack_output = x
        encoder_output = layers.layer_normalize(x)

        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

        return encoder_output, encoder_decoder_attention_bias
