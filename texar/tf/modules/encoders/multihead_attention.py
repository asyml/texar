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
Transformer encoders with multi-head self attention.
"""

import tensorflow as tf

from texar.tf.core.layers import get_initializer
from texar.tf.modules.encoders.encoder_base import EncoderBase
from texar.tf.utils.mode import is_train_mode
from texar.tf.utils.shapes import transpose_batch_time


__all__ = [
    "MultiheadAttentionEncoder"
]


class MultiheadAttentionEncoder(EncoderBase):
    r"""Multi-head Attention Encoder

    Args:
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.
    """
    def __init__(self, hparams=None):
        super().__init__(hparams=hparams)
        use_bias = self._hparams.use_bias
        initializer = get_initializer(self._hparams.initializer)

        self.Q_dense = tf.keras.layers.Dense(self._hparams.num_units,
                                             use_bias=use_bias,
                                             kernel_initializer=initializer,
                                             bias_initializer=initializer)
        self.K_dense = tf.keras.layers.Dense(self._hparams.num_units,
                                             use_bias=use_bias,
                                             kernel_initializer=initializer,
                                             bias_initializer=initializer)
        self.V_dense = tf.keras.layers.Dense(self._hparams.num_units,
                                             use_bias=use_bias,
                                             kernel_initializer=initializer,
                                             bias_initializer=initializer)
        self.O_dense = tf.keras.layers.Dense(self._hparams.output_dim,
                                             use_bias=use_bias,
                                             kernel_initializer=initializer,
                                             bias_initializer=initializer)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "initializer": None,
                "num_heads": 8,
                "output_dim": 512,
                "num_units": 512,
                "dropout_rate": 0.1,
                "use_bias": False,
                "name": "multihead_attention"
            }

        Here:

        `"initializer"`: dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.
            See :func:`~texar.tf.core.get_initializer` for details.

        `"num_heads"`: int
            Number of heads for attention calculation.

        `"output_dim"`: int
            Output dimension of the returned tensor.

        `"num_units"`: int
            Hidden dimension of the unsplitted attention space.
            Should be devisible by `num_heads`.

        `"dropout_rate"`: : float
            Dropout rate in the attention.

        `"use_bias"`: bool
            Use bias when projecting the key, value and query.

        `"name"`: str
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

    def __call__(self, queries, memory, memory_attention_bias,
             cache=None, mode=None):
        r"""Encodes the inputs.

        Args:
            queries: A 3d tensor with shape of
                ``[batch, length_query, depth_query]``.
            memory: A 3d tensor with shape of
                ``[batch, length_key, depth_key]``.
            memory_attention_bias: A 3d tensor with shape of
                ``[batch, length_key, num_units]``.
            cache: Memory cache only when inferring the sentence from scratch.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, including
                `TRAIN`, `EVAL`, and `PREDICT`.

        Returns:
            A Tensor of shape ``[batch_size, max_time, dim]`` containing the
            encoded vectors.
        """
        return super().__call__([queries, memory, memory_attention_bias],
                                cache, mode)

    def call(self, inputs, cache, mode):
        r"""Encodes the inputs.
        """
        queries, memory, memory_attention_bias = inputs
        num_heads = self._hparams.num_heads
        num_units = self._hparams.num_units
        if num_units % num_heads:
            raise ValueError("Value depth (%d) must be divisible by "
                             "the number of attention heads (%d)."
                             % (num_units, num_heads))

        def _update_and_return(layer, key):
            if memory is None:
                # Self Attention
                out = layer(queries)

                if cache is not None:
                    # 'decoder self attention when dynamic decoding'
                    key = 'self_{}'.format(key)
                    res = cache[key]
                    if isinstance(res, tf.TensorArray):
                        # inference-like decoding
                        # TODO(zhiting): This writing op may cause a bug
                        # on CPU--it looks the two TensorArray
                        # cache['self_keys'] and cache['self_values']
                        # will mix up starting from certain step, causing
                        # shape mismatch. This op looks fine on GPU.
                        res = res.write(res.size(), tf.squeeze(out, axis=[1]))
                        out = transpose_batch_time(res.stack())
                    else:
                        # normal decoding
                        res = tf.concat([res, out], axis=1)
                        out = res
                    cache[key] = res

            else:
                # encoder decoder attention
                if cache is not None:
                    key = 'memory_{}'.format(key)
                    res = cache[key]
                    if isinstance(res, tf.TensorArray):
                        # inference-like decoding
                        size = res.size()
                        false_fn = lambda: transpose_batch_time(res.stack())
                    else:
                        # normal decoding
                        size = res.shape[1]
                        false_fn = lambda: res

                    if size == 0:
                        out = layer(memory)
                    else:
                        out = false_fn()
                else:
                    out = layer(memory)

            return out

        Q = self.Q_dense(queries)
        K = _update_and_return(self.K_dense, 'keys')
        V = _update_and_return(self.V_dense, 'values')

        Q_ = self._split_heads(Q)
        K_ = self._split_heads(K)
        V_ = self._split_heads(V)
        # [batch_size, num_heads, seq_length, memory_depth]
        key_depth_per_head = num_units // num_heads
        Q_ *= key_depth_per_head**-0.5

        logits = tf.matmul(Q_, K_, transpose_b=True)
        if memory_attention_bias is not None:
            logits += memory_attention_bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        if is_train_mode(mode):
            weights = tf.nn.dropout(weights,
                                    rate=self._hparams.dropout_rate)
        outputs = tf.matmul(weights, V_)

        outputs = self._combine_heads(outputs)
        outputs = self.O_dense(outputs)
        # (batch_size, length_query, output_dim)

        return outputs

    def _split_heads(self, x):
        r"""Split channels (dimension 2) into multiple heads,
        becomes dimension 1).

        Must ensure `x.shape[-1]` can be divided by num_heads.
        """
        depth = x.shape.as_list()[-1]
        splitted_x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1],
                                    self._hparams.num_heads,
                                    depth // self._hparams.num_heads])
        return tf.transpose(splitted_x, [0, 2, 1, 3])

    def _combine_heads(self, x):
        r"""
        Args:
            x: A Tensor of shape ``[batch, num_heads, seq_len, dim]``

        Returns:
            A Tensor of shape ``[batch, seq_len, num_heads * dim]``
        """
        t = tf.transpose(x, [0, 2, 1, 3])  # [batch, seq_len, num_heads, dim]
        num_heads, dim = t.shape.as_list()[-2:]
        assert num_heads == self._hparams.num_heads
        return tf.reshape(t, [tf.shape(t)[0], tf.shape(t)[1], num_heads * dim])
