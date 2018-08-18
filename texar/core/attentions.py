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
"""Attentions specific to Transformer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf

from texar import context

# pylint: disable=too-many-arguments, invalid-name, no-member

__all__ = [
    'attention_bias_lower_triangle',
    'attention_bias_ignore_padding',
    'attention_bias_local',
    'multihead_attention',
]

def attention_bias_lower_triangle(length):
    """Create an bias tensor to be added to attention logits.
    Allows a query to attend to all positions up to and including its own.

    Args:
        length: a scalar.

    Returns:
        a `Tensor` with shape [1, 1, length, length].
    """
    return attention_bias_local(length, -1, 0)

def attention_bias_local(length, max_backward, max_forward):
    """Create an bias tensor to be added to attention logits.
    A position may attend to positions at most max_distance from it,
    forward and backwards.

    This does not actually save any computation.

    Args:
        length: int
        max_backward: int, maximum distance backward to attend. Negative values
            indicate unlimited.
        max_forward: int, maximum distance forward to attend. Negative values
            indicate unlimited.

    Returns:
        a `Tensor` with shape [1, 1, length, length].
        [batch_size, num_heads, queri_len, queri_len]
    """
    band = _ones_matrix_band_part(
        length,
        length,
        max_backward,
        max_forward,
        out_shape=[1, 1, length, length])
    return -1e18 * (1.0 - band)

def attention_bias_ignore_padding(memory_padding):
    """Create an bias tensor to be added to attention logits.

    Args:
        memory_padding: a float `Tensor` with shape [batch, memory_length].

    Returns:
        a `Tensor` with shape [batch, 1, 1, memory_length].
        each dim corresponding to batch_size, num_heads, queries_len,
        memory_length
    """
    ret = memory_padding * -1e18
    return tf.expand_dims(tf.expand_dims(ret, axis=1), axis=1)

def multihead_attention(queries,
                        memory_attention_bias=None,
                        memory=None,
                        num_heads=8,
                        num_units=None,
                        dropout_rate=0,
                        cache=None,
                        scope='multihead_attention'):
    """Applies multihead attention.

    Args:
        queries: A 3d tensor with shape of [batch, length_query, depth_query].
        keys: A 3d tensor with shape of [batch, length_key, depth_key].
        num_units: A scalar indicating the attention size,
            equals to depth_query if not given.
        dropout_rate: A floating point number.
        num_heads: An int. Number of heads with calculating attention.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

    Returns:
        A 3d tensor with shape of (batch, length_query, num_units)
    """
    #pylint: disable=too-many-locals
    with tf.variable_scope(scope):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        if num_units % num_heads != 0:
            raise ValueError("Value depth (%d) must be divisible by the number"
                             "of attention heads (%d)." % (\
                            num_units, num_heads))
        if memory is None:
            #'self attention'
            Q = tf.layers.dense(queries, num_units, use_bias=False, name='q')
            K = tf.layers.dense(queries, num_units, use_bias=False, name='k')
            V = tf.layers.dense(queries, num_units, use_bias=False, name='v')
            if cache is not None:
                # 'decoder self attention when dynamic decoding'
                K = tf.concat([cache['self_keys'], K], axis=1)
                V = tf.concat([cache['self_values'], V], axis=1)
                cache['self_keys'] = K
                cache['self_values'] = V
        else:
            # 'encoder decoder attention'
            Q = tf.layers.dense(queries, num_units, use_bias=False, name='q')
            if cache is not None:
                K, V = tf.cond(
                    tf.equal(tf.shape(cache["memory_keys"])[1], 0),
                    true_fn=lambda: \
                        [tf.layers.dense(memory, num_units, \
                            use_bias=False, name='k'), \
                        tf.layers.dense(memory, num_units, \
                            use_bias=False, name='v')],
                    false_fn=lambda: \
                        [cache["memory_keys"], cache["memory_values"]])
            else:
                K, V = [tf.layers.dense(memory, num_units, \
                            use_bias=False, name='k'),
                        tf.layers.dense(memory, num_units, \
                            use_bias=False, name='v')]

        Q_ = _split_heads(Q, num_heads)
        K_ = _split_heads(K, num_heads)
        V_ = _split_heads(V, num_heads)
        #[batch_size, num_heads, seq_length, memory_depth]
        key_depth_per_head = num_units // num_heads
        Q_ *= key_depth_per_head**-0.5

        logits = tf.matmul(Q_, K_, transpose_b=True)
        if memory_attention_bias is not None:
            logits += memory_attention_bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        weights = tf.layers.dropout(weights, \
            rate=dropout_rate, training=context.global_mode_train())
        outputs = tf.matmul(weights, V_)

        outputs = _combine_heads(outputs)
        outputs = tf.layers.dense(outputs, num_units,\
            use_bias=False, name='output_transform')
        #(batch_size, length_query, attention_depth)
    return outputs


def _split_heads(x, num_heads):
    """Split channels (dimension 2) into multiple heads, becomes dimension 1).
    Must ensure `x.shape[-1]` can be deviced by num_heads.any
    """
    depth = x.get_shape()[-1]
    splitted_x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], \
        num_heads, depth // num_heads])
    return tf.transpose(splitted_x, [0, 2, 1, 3])


def _combine_heads(x):
    """
    Args:
        x: A Tensor of shape `[batch, num_heads, seq_len, dim]`

    Returns:
        A Tensor of shape `[batch, seq_len, num_heads * dim]`
    """
    t = tf.transpose(x, [0, 2, 1, 3]) #[batch, seq_len, num_heads, dim]
    num_heads, dim = t.get_shape()[-2:]
    return tf.reshape(t, [tf.shape(t)[0], tf.shape(t)[1], num_heads*dim])


def _ones_matrix_band_part(rows, cols, num_lower, num_upper, out_shape=None):
    """Matrix band part of ones.
    """
    if all([isinstance(el, int) for el in [rows, cols, num_lower, num_upper]]):
    # Needed info is constant, so we construct in numpy
        if num_lower < 0:
            num_lower = rows - 1
        if num_upper < 0:
            num_upper = cols - 1
        lower_mask = np.tri(cols, rows, num_lower).T
        upper_mask = np.tri(rows, cols, num_upper)
        band = np.ones((rows, cols)) * lower_mask * upper_mask
        if out_shape:
            band = band.reshape(out_shape)
        band = tf.constant(band, tf.float32)
    else:
        band = tf.matrix_band_part(tf.ones([rows, cols]),
                                   tf.cast(num_lower, tf.int64),
                                   tf.cast(num_upper, tf.int64))
        if out_shape:
            band = tf.reshape(band, out_shape)
    return band
