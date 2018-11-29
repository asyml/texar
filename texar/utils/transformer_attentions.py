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
]

def attention_bias_lower_triangle(length, bias_value=-1e18):
    """Create an bias tensor to be added to attention logits.
    Allows a query to attend to all positions up to and including its own.

    Args:
        length: a scalar.

    Returns:
        a `Tensor` with shape [1, 1, length, length].
    """
    return attention_bias_local(length, -1, 0, bias_value)

def attention_bias_local(length, max_backward, max_forward, bias_value=-1e18):
    """Create an bias tensor to be added to attention logits.
    A position may attend to positions at most max_distance from it,
    forward and backwards.

    This does not actually save any computation.

    Args:
        length: int
        max_backward: int, maximum distance backward to attend. Negative
        values indicate unlimited.
        max_forward: int, maximum distance forward to attend. Negative
        values indicate unlimited.

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
    return bias_value * (1.0 - band)

def attention_bias_ignore_padding(memory_padding, bias_value=-1e18):
    """Create an bias tensor to be added to attention logits.

    Args:
        memory_padding: a float `Tensor` with shape [batch, memory_length].

    Returns:
        a `Tensor` with shape [batch, 1, 1, memory_length].
        each dim corresponding to batch_size, num_heads, queries_len,
        memory_length
    """
    ret = memory_padding * bias_value
    return tf.expand_dims(tf.expand_dims(ret, axis=1), axis=1)

def _ones_matrix_band_part(rows, cols, num_lower, num_upper,
    out_shape=None):
    """Matrix band part of ones.
    """
    if all([isinstance(el, int) for el in [rows, cols, num_lower,
        num_upper]]):
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
