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
Utility functions related to tensor shapes.
"""

import numpy as np
import tensorflow as tf


__all__ = [
    "transpose_batch_time",
    "get_batch_size",
    "get_rank",
    "mask_sequences",
]


def transpose_batch_time(inputs):
    r"""Transposes inputs between time-major and batch-major.

    Args:
        inputs: A Tensor of shape ``[batch_size, max_time, ...]`` (batch-major)
            or ``[max_time, batch_size, ...]`` (time-major), or a (possibly
            nested) tuple of such elements.

    Returns:
        A (possibly nested tuple of) Tensor with transposed batch and
        time dimensions of inputs.
    """
    rank = get_rank(inputs)
    perm = [1, 0] + [i for i in range(2, rank)]
    return tf.transpose(inputs, perm=perm)


def get_batch_size(tensor):
    r"""Returns an  ``int`` representing the batch size, i.e.,
    the size of the 1st dimension of :attr:`tensor`.
    """
    return tensor.shape[0]


def get_rank(tensor):
    r"""Returns the tensor rank as a python ``int``. The input tensor can also
    be a python array.

    Args:
        tensor: A Tensor or python array.

    Returns:
        A python ``int`` representing the rank of :attr:`tensor`. Returns
        `None` if the rank cannot be determined.
    """
    if tf.is_tensor(tensor):
        rank = len(tensor.shape)
    else:
        array = np.asarray(tensor)
        rank = array.ndim
    return rank


def mask_sequences(sequence,
                   sequence_length,
                   dtype=None,
                   time_major=False):
    r"""Masks out sequence entries that are beyond the respective sequence
    lengths. Masks along the time dimension.

    :attr:`sequence` and :attr:`sequence_length` can either be python
    arrays or Tensors, respectively. If both are python arrays (or None), the
    return will be a python array as well.

    Args:
        sequence: A Tensor or python array of sequence values.
            If ``time_major==False`` (default), this must be a Tensor of shape
            ``[batch_size, max_time, ...]``. The batch and time dimension is
            exchanged if ``time_major==True``.
        sequence_length: A Tensor or python array of shape ``[batch_size]``.
            Time steps beyond the respective sequence lengths will be
            made zero.
        dtype (dtype): Type of :attr:`sequence`. If `None`, infer from
            :attr:`sequence` automatically.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`sequence` must have shape
            ``[max_time, batch_size, ...]``.
            If `False` (default), :attr:`sequence` must have
            shape ``[batch_size, max_time, ...]``.

    Returns:
        The masked sequence, i.e., a Tensor or python array of the same shape
        as :attr:`sequence` but with masked-out entries (set to zero).

        If both :attr:`sequence` and :attr:`sequence_length` are python
        arrays, the returned value is a python array as well.
    """
    if not tf.is_tensor(sequence):
        sequence = tf.convert_to_tensor(sequence, dtype=dtype)

    rank = get_rank(sequence)
    if rank < 2:
        raise ValueError("`sequence` must be 2D or higher order.")

    if time_major:
        sequence = transpose_batch_time(sequence)
    max_time = sequence.shape[1]
    if dtype is None:
        dtype = sequence.dtype
    mask = tf.sequence_mask(
        tf.cast(sequence_length, tf.int32), max_time, dtype=dtype)
    for _ in range(2, rank):
        mask = tf.expand_dims(mask, axis=-1)
    sequence = sequence * mask
    if time_major:
        sequence = transpose_batch_time(sequence)
    return sequence
