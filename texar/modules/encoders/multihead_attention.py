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
from texar.modules.encoders.encoder_base import EncoderBase
from texar.utils.shapes import shape_list
from texar.utils.mode import is_train_mode

# pylint: disable=too-many-locals, invalid-name, arguments-differ
# pylint: disable=too-many-arguments

__all__ = [
    "MultiheadAttentionEncoder"
]

class MultiheadAttentionEncoder(EncoderBase):
    """Multihead Attention Encoder

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
        use_bias = self._hparams.use_bias

        with tf.variable_scope(self.variable_scope):
            if self._hparams.initializer:
                tf.get_variable_scope().set_initializer(
                    layers.get_initializer(self._hparams.initializer))

            self.Q_dense = tf.layers.Dense(self._hparams.num_units,
                                           use_bias=use_bias,
                                           name='query')
            self.K_dense = tf.layers.Dense(self._hparams.num_units,
                                           use_bias=use_bias,
                                           name='key')
            self.V_dense = tf.layers.Dense(self._hparams.num_units,
                                           use_bias=use_bias,
                                           name='value')
            self.O_dense = tf.layers.Dense(self._hparams.output_dim,
                                           use_bias=use_bias,
                                           name='output')
    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "initializer": None,
                'num_heads': 8,
                'output_dim': 512,
                'num_units': 512,
                'dropout_rate': 0.1,
                'use_bias': False,
                "name": "multihead_attention"
            }

        Here:

        "initializer" : dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.
            See :func:`~texar.core.get_initializer` for details.

        "num_heads" : int
            Number of heads for attention calculation.

        "output_dim" : int
            Output dimension of the returned tensor.

        "num_units" : int
            Hidden dimension of the unsplitted attention space.
            Should be devisible by `num_heads`.

        "dropout_rate: : float
            Dropout rate in the attention.

        "use_bias": bool
            Use bias when projecting the key, value and query.

        "name" : str
            Name of the module.
        """
        return {
            'initializer': None,
            'num_heads': 8,
            'output_dim': 512,
            'num_units': 512,
            'dropout_rate': 0.1,
            'use_bias': False,
            "name": "multihead_attention",
        }

    def _build(self, queries, memory, memory_attention_bias,
               cache=None, mode=None):
        """Encodes the inputs.

        Args:
            queries: A 3d tensor with shape of [batch, length_query,
                depth_query].
            memory: A 3d tensor with shape of [batch, length_key, depth_key].
            memory_attention_bias: A 3d tensor with shape of
                [batch, length_key, num_units].
            cache: Memory cache only when inferencing the sentence from sractch.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, including
                `TRAIN`, `EVAL` and `PREDICT`. Controls dropout mode.
                If `None` (default), :func:`texar.global_mode` is used.

        Returns:
            A Tensor of shape `[batch_size, max_time, dim]` containing the
            encoded vectors.
        """

        with tf.variable_scope(self.variable_scope):
            num_heads = self._hparams.num_heads
            num_units = self._hparams.num_units
            if num_units % num_heads:
                raise ValueError("Value depth (%d) must be divisible by "
                                 "the number of attention heads (%d)." %(\
                                 num_units, num_heads))
            if memory is None:
                # Self Attention
                Q = self.Q_dense(queries)
                K = self.K_dense(queries)
                V = self.V_dense(queries)

                if cache is not None:
                    # 'decoder self attention when dynamic decoding'
                    K = tf.concat([cache['self_keys'], K], axis=1)
                    V = tf.concat([cache['self_values'], V], axis=1)
                    cache['self_keys'] = K
                    cache['self_values'] = V
            else:
                # encoder decoder attention
                Q = self.Q_dense(queries)
                if cache is not None:
                    K, V = tf.cond(
                        tf.equal(tf.shape(cache["memory_keys"])[1], 0),
                        true_fn=lambda: \
                            [self.K_dense(memory), self.V_dense(memory)],
                        false_fn=lambda: \
                            [cache["memory_keys"], cache["memory_values"]])
                else:
                    K, V = [self.K_dense(memory), self.V_dense(memory)]

            Q_ = self._split_heads(Q)
            K_ = self._split_heads(K)
            V_ = self._split_heads(V)
            #[batch_size, num_heads, seq_length, memory_depth]
            key_depth_per_head = num_units // num_heads
            Q_ *= key_depth_per_head**-0.5

            logits = tf.matmul(Q_, K_, transpose_b=True)
            if memory_attention_bias is not None:
                logits += memory_attention_bias
            weights = tf.nn.softmax(logits, name="attention_weights")
            weights = tf.layers.dropout(weights,
                                        rate=self._hparams.dropout_rate,
                                        training=is_train_mode(mode))
            outputs = tf.matmul(weights, V_)

            outputs = self._combine_heads(outputs)
            outputs = self.O_dense(outputs)
            #(batch_size, length_query, output_dim)

        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

        return outputs

    def _split_heads(self, x):
        """Split channels (dimension 2) into multiple heads,
        becomes dimension 1).

        Must ensure `x.shape[-1]` can be deviced by num_heads
        """
        depth = shape_list(x)[-1]
        splitted_x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], \
            self._hparams.num_heads, depth // self._hparams.num_heads])
        return tf.transpose(splitted_x, [0, 2, 1, 3])

    def _combine_heads(self, x):
        """
        Args:
            x: A Tensor of shape `[batch, num_heads, seq_len, dim]`

        Returns:
            A Tensor of shape `[batch, seq_len, num_heads * dim]`
        """
        t = tf.transpose(x, [0, 2, 1, 3]) #[batch, seq_len, num_heads, dim]
        num_heads, dim = shape_list(t)[-2:]
        assert num_heads == self._hparams.num_heads
        return tf.reshape(t, [tf.shape(t)[0], tf.shape(t)[1], num_heads*dim])
