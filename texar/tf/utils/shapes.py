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
Utility functions related to tensor shapes.
"""

# pylint: disable=no-name-in-module, protected-access, no-member, invalid-name

import numpy as np

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn
from tensorflow.python.framework import ops

__all__ = [
    "transpose_batch_time",
    "get_batch_size",
    "get_rank",
    "mask_sequences",
    "_mask_sequences_tensor",
    "_mask_sequences_py",
    "reduce_with_weights",
    "flatten",
    "shape_list",
    "pad_and_concat",
    "varlength_concat",
    "varlength_concat_py",
    "varlength_roll"
]


def transpose_batch_time(inputs):
    """Transposes inputs between time-major and batch-major.

    Args:
        inputs: A Tensor of shape `[batch_size, max_time, ...]` (batch-major)
            or `[max_time, batch_size, ...]` (time-major), or a (possibly
            nested) tuple of such elements.

    Returns:
        A (possibly nested tuple of) Tensor with transposed batch and
        time dimensions of inputs.
    """
    flat_input = nest.flatten(inputs)
    flat_input = [ops.convert_to_tensor(input_) for input_ in flat_input]
    # pylint: disable=protected-access
    flat_input = [rnn._transpose_batch_time(input_) for input_ in flat_input]
    return nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)


def get_batch_size(tensor):
    """Returns a unit `Tensor` representing the batch size, i.e.,
    the size of the 1st dimension of :attr:`tensor`.
    """
    return tf.shape(tensor)[0]


def get_rank(tensor):
    """Returns the tensor rank as a python `int`. The input tensor can also be
    a python array.

    Args:
        tensor: A Tensor or python array.

    Returns:
        A python `int` representing the rank of :attr:`tensor`. Returns
        `None` if the rank cannot be determined.
    """
    if tf.contrib.framework.is_tensor(tensor):
        shape = tensor.shape
        try:
            rank = len(shape.as_list())
        except ValueError:  # when `shape==TensorShape(None)`
            rank = None
    else:
        array = np.asarray(tensor)
        rank = array.ndim
    return rank


def mask_sequences(sequence,
                   sequence_length,
                   dtype=None,
                   time_major=False,
                   tensor_rank=2):
    """Masks out sequence entries that are beyond the respective sequence
    lengths. Masks along the time dimension.

    :attr:`sequence` and :attr:`sequence_length` can either be python
    arrays or Tensors, respectively. If both are python arrays (or None), the
    return will be a python array as well.

    Args:
        sequence: A Tensor or python array of sequence values.
            If `time_major==False` (default), this must be a Tensor of shape
            `[batch_size, max_time, ...]`. The batch and time dimension is
            exchanged if `time_major==True`.
        sequence_length: A Tensor or python array of shape `[batch_size]`.
            Time steps beyond the respective sequence lengths will be
            made zero.
        dtype (dtype): Type of :attr:`sequence`. If `None`, infer from
            :attr:`sequence` automatically.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`sequence` must have shape
            `[max_time, batch_size, ...]`.
            If `False` (default), :attr:`sequence` must have
            shape `[batch_size, max_time, ...]`.
        tensor_rank (int): The number of dimensions of :attr:`sequence`.
            Default is 2, i.e., :attr:`sequence` is a 2D Tensor consisting
            of batch and time dimensions. Ignored if both :attr:`sequence`
            and :attr:`sequence_length` are python arrays.

    Returns:
        The masked sequence, i.e., a Tensor or python array of the same shape
        as :attr:`sequence` but with masked-out entries (set to zero).

        If both :attr:`sequence` and :attr:`sequence_length` are python
        arrays, the returned value is a python array as well.
    """
    is_tensor = tf.contrib.framework.is_tensor
    if is_tensor(sequence) or is_tensor(sequence_length):
        return _mask_sequences_tensor(
            sequence, sequence_length, dtype, time_major, tensor_rank)
    else:
        return _mask_sequences_py(
            sequence, sequence_length, dtype, time_major)


def _mask_sequences_tensor(sequence,
                           sequence_length,
                           dtype=None,
                           time_major=False,
                           tensor_rank=2):
    """Masks out sequence entries that are beyond the respective sequence
    lengths. Masks along the time dimension.

    Args:
        sequence: A Tensor of sequence values.

            If `time_major=False` (default), this must be a Tensor of shape:
                `[batch_size, max_time, d_2, ..., d_rank]`, where the rank of
                the Tensor is specified with :attr:`tensor_rank`.

            If `time_major=True`, this must be a Tensor of shape:
                `[max_time, batch_size, d_2, ..., d_rank].`
        sequence_length: A Tensor of shape `[batch_size]`. Time steps beyond
            the respective sequence lengths will be made zero.
        dtype (dtype): Type of :attr:`sequence`. If `None`, infer from
            :attr:`sequence` automatically.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`sequence` must have shape
            `[max_time, batch_size, d_2, ..., d_rank]`.
            If `False` (default), :attr:`sequence` must have
            shape `[batch_size, max_time, d_2, ..., d_rank]`.
        tensor_rank (int): The number of dimensions of :attr:`sequence`.
            Default is 2, i.e., :attr:`sequence` is a 2D Tensor consisting
            of batch and time dimensions.

    Returns:
        The masked sequence, i.e., a Tensor of the same shape as
        :attr:`sequence` but with masked-out entries (set to zero).
    """
    if tensor_rank is None:
        tensor_rank = 2
    if tensor_rank < 2:
        raise ValueError(
            "tensor_rank must be > 2. Got tensor_rank = {}".format(tensor_rank))
    if time_major:
        sequence = rnn._transpose_batch_time(sequence)
    max_time = tf.cast(tf.shape(sequence)[1], tf.int32)
    if dtype is None:
        dtype = sequence.dtype
    mask = tf.sequence_mask(
        tf.cast(sequence_length, tf.int32), max_time, dtype=dtype)
    for _ in range(2, tensor_rank):
        mask = tf.expand_dims(mask, axis=-1)
    sequence = sequence * mask
    if time_major:
        sequence = rnn._transpose_batch_time(sequence)
    return sequence


def _mask_sequences_py(sequence,
                       sequence_length,
                       dtype=None,
                       time_major=False):
    """Masks out sequence entries that are beyond the respective sequence
    lengths. Masks along the time dimension.

    This is the numpy version of :func:`texar.tf.utils.mask_sequences`.

    Args:
        sequence: An python array of sequence values.

            If `time_major=False` (default), this must be an array of shape:
                `[batch_size, max_time, ...]`

            If `time_major=True`, this must be a Tensor of shape:
                `[max_time, batch_size, ...].`
        sequence_length: An array of shape `[batch_size]`. Time steps beyond
            the respective sequence lengths will be made zero.
        dtype (dtype): Type of :attr:`sequence`. If `None`, infer from
            :attr:`sequence` automatically.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`sequence` must have shape
            `[max_time, batch_size, ...]`.
            If `False` (default), :attr:`sequence` must have
            shape `[batch_size, max_time, ...]`.

    Returns:
        The masked sequence, i.e., an array of the same shape as
        :attr:`sequence` but with masked-out entries (set to zero).
    """
    sequence = np.array(sequence)
    sequence_length = np.array(sequence_length)

    rank = sequence.ndim
    if rank < 2:
        raise ValueError("`sequence` must be 2D or higher order.")
    batch_size = sequence.shape[0]
    max_time = sequence.shape[1]
    dtype = dtype or sequence.dtype

    if time_major:
        sequence = np.transpose(sequence, axes=[1, 0, 2])

    steps = np.tile(np.arange(max_time), [batch_size, 1])
    mask = np.asarray(steps < sequence_length[:, None], dtype=dtype)
    for _ in range(2, rank):
        mask = np.expand_dims(mask, -1)

    sequence = sequence * mask

    if time_major:
        sequence = np.transpose(sequence, axes=[1, 0, 2])

    return sequence


def reduce_with_weights(tensor,
                        weights=None,
                        average_across_batch=True,
                        average_across_remaining=False,
                        sum_over_batch=False,
                        sum_over_remaining=True,
                        tensor_rank=None):
    """Weights and reduces tensor.

    Args:
        tensor: A Tensor to weight and reduce, of shape
            `[batch_size, ...]`.
        weights (optional): A Tensor of the same shape and dtype with
            :attr:`tensor`. For example, this is can be a 0-1 tensor
            for masking values of :attr:`tensor``.
        average_across_batch (bool): If set, average the tensor across the
            batch dimension. Must not set `average_across_batch`'
            and `sum_over_batch` at the same time.
        average_across_remaining (bool): If set, average the
            tensor across the
            remaining dimensions. Must not set `average_across_remaining`'
            and `sum_over_remaining` at the same time.
            If :attr:`weights` is given, this is a weighted average.
        sum_over_batch (bool): If set, sum the tensor across the
            batch dimension. Must not set `average_across_batch`
            and `sum_over_batch` at the same time.
        sum_over_remaining (bool): If set, sum the tensor
            across the
            remaining dimension. Must not set `average_across_remaining`
            and `sum_over_remaining` at the same time.
            If :attr:`weights` is given, this is a weighted sum.
        tensor_rank (int, optional): The number of dimensions of
            :attr:`tensor`. If not given, inferred from :attr:`tensor`
            automatically.

    Returns:
        A Tensor.

    Example:
        .. code-block:: python

            x = tf.constant([[10, 10, 2, 2],
                             [20, 2, 2, 2]])
            mask = tf.constant([[1, 1, 0, 0],
                                [1, 0, 0, 0]])

            z = reduce_with_weights(x, weights=mask)
            # z == 20
            # (all 2 in x are masked)
    """
    if tensor_rank is None:
        tensor_rank = get_rank(tensor)
    if tensor_rank is None:
        raise ValueError('Unable to infer the rank of `tensor`. '
                         'Please set `tensor_rank` explicitly.')

    if weights is not None:
        tensor = tensor * weights

    if tensor_rank > 1:
        if average_across_remaining and sum_over_remaining:
            raise ValueError("Only one of `average_across_remaining` and "
                             "`sum_over_remaining` can be set.")
        if average_across_remaining:
            if weights is None:
                tensor = tf.reduce_mean(tensor, axis=np.arange(1, tensor_rank))
            else:
                tensor = tf.reduce_sum(tensor, axis=np.arange(1, tensor_rank))
                weights = tf.reduce_sum(weights, axis=np.arange(1, tensor_rank))
                tensor = tensor / weights
        elif sum_over_remaining:
            tensor = tf.reduce_sum(tensor, axis=np.arange(1, tensor_rank))

    if average_across_batch and sum_over_batch:
        raise ValueError("Only one of `average_across_batch` and "
                         "`sum_over_batch` can be set.")
    if sum_over_batch:
        tensor = tf.reduce_sum(tensor, axis=[0])
    elif average_across_batch:
        tensor = tf.reduce_mean(tensor, axis=[0])

    return tensor


def flatten(tensor, preserve_dims, flattened_dim=None):
    """Flattens a tensor whiling keeping several leading dimensions.

    :attr:`preserve_dims` must < tensor's rank

    Args:
        tensor: A Tensor to flatten.
        preserve_dims (int): The number of leading dimensions to preserve.
        flatterned_dim (int, optional): The size of the resulting flattened
            dimension. If not given, infer automatically, which can cause
            a statically unknown dimension size.

    Returns:
        A Tensor with rank :attr:`perserve_dims` + 1.

    Example:
        .. code-block:: python

            x = tf.ones(shape=[d_1, d_2, d_3, d_4])
            y = flatten(x, 2) # y.shape == [d_1, d_2, d_3 * d_4]
    """
    if flattened_dim is None:
        flattened_dim = -1
    shape = tf.concat([tf.shape(tensor)[:preserve_dims], [flattened_dim]],
                      axis=0)
    tensor_ = tf.reshape(tensor, shape)
    return tensor_


def shape_list(x):
    r"""Returns **static** shape of the input Tensor whenever possible.

    Args:
        x: A Tensor.

    Returns:
        - If the rank of `x` is unknown, returns the dynamic shape
          ``tf.shape(x)``

        - Otherwise, returns a list of dims, each of which is either an `int`
          whenever it can be statically determined, or a scalar Tensor
          otherwise.
    """
    x = tf.convert_to_tensor(x)
    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)
    static = x.get_shape().as_list()
    shape = tf.shape(x)
    ret = []
    for i, dim in enumerate(static):
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def pad_and_concat(values, axis, rank=None, pad_axis=None,
                   pad_constant_values=0):
    """Concats tensors along one dimension. Pads each of other dimensions of
    the tensors to the corresponding maximum size if necessary.

    Args:
        values: A list of Tensors of the same rank.
        axis (int): A Python int. Dimension along which to concatenate.
        rank (int, optional): Rank of the tensors. If `None`, inferred
            automatically from :attr:`values`.
        pad_axis (int or list, optional): A Python int or a list of int.
            Dimensions to pad. Paddings are only added to the end of
            corresponding dimensions. If `None`, all dimensions except the
            :attr:`axis` dimension are padded.
        pad_constant_values: The scalar pad value to use. Must be same type
            as the tensors.

    Returns:
        A `Tensor` resulting from padding and concatenation of the input
        tensors.

    Raises:
        ValueError: If :attr:`rank` is `None` and cannot be inferred from
            :attr:`values`.


    Example:

        .. code-block:: python

            a = tf.ones([1, 2])
            b = tf.ones([2, 3])

            c = pad_and_concat([a,b], 0)
            # c.shape == [3, 3]
            # c == [[1, 1, 0],
            #       [1, 1, 1],
            #       [1, 1, 1]]

            d = pad_and_concat([a,b], 1)
            # d.shape == [2, 5]
            # d == [[1, 1, 1, 1, 1]
            #       [0, 0, 1, 1, 1]]
    """
    if rank is None:
        for value in values:
            rank = get_rank(value)
            if rank is not None:
                break
    if rank is None:
        raise ValueError('Cannot determine the rank of the tensors')

    def _pad_to_size(value, axis_, size):
        """Pads the :attr:`axis_` of a tensor :attr:`value` to the given
        :attr:`size`. Only pads to the end.

        Args:
            value: A Tensor.
            axis_: A Python int.
            size: A scalar int Tensor or Python int.
        """
        paddings = np.zeros([rank, 2], dtype=np.int32)
        paddings[axis_, 1] = 1
        paddings = paddings * (size - tf.shape(value)[axis_])
        return tf.pad(value, paddings, mode='CONSTANT',
                      constant_values=pad_constant_values)

    if pad_axis is None:
        pad_axis = [r for r in range(rank) if r != axis]

    pad_axis = pad_axis if isinstance(pad_axis, (list, tuple)) else [pad_axis]

    for pa in pad_axis:
        max_dim_size = tf.reduce_max([tf.shape(v)[pa] for v in values])
        for i, v in enumerate(values):
            values[i] = _pad_to_size(v, pa, max_dim_size)

    return tf.concat(values, axis)


def varlength_concat(x, y, x_length, dtype=None, tensor_rank=None):
    """Concatenates rows of `x` and `y` where each row of
    `x` has a variable length.

    Both `x` and `y` are of numeric dtypes, such as `tf.int32` and `tf.float32`,
    with mask value `0`. The two tensors must be of the same dtype.

    Args:
        x: A tensor of shape `[batch_size, x_dim_2, other_dims]`.
        y: A tensor of shape `[batch_size, y_dim_2, other_dims]`.
            All dimensions except the 2nd dimension must be the same
            with those of `x`.
        x_length: A 1D int tensor of shape `[batch_size]` containing
            the length of each `x` row.
            Elements beyond the respective lengths will be
            made zero.
        dtype: Type of :attr:`x`. If `None`, inferred from
            :attr:`x` automatically.
        tensor_rank (int, optional): The number of dimensions of
            :attr:`x`. If not given, inferred from :attr:`x`
            automatically.

    Returns:
        A Tensor of shape `[batch_size, x_dim_2 + y_dim_2, other_dims]`.

    Example:
        .. code-block:: python

            x = tf.constant([[1, 1, 0, 0],
                             [1, 1, 1, 0]])
            x_length = [2, 3]
            y = tf.constant([[2, 2, 0],
                             [2, 2, 2]])

            out = varlength_concat(x, y, x_length)
            # out = [[1, 1, 2, 2, 0, 0, 0]
            #        [1, 1, 1, 2, 2, 2, 0]]
    """
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    x_length = tf.convert_to_tensor(x_length)

    if tensor_rank is None:
        tensor_rank = get_rank(x) or get_rank(y)
    if tensor_rank is None:
        raise ValueError('Unable to infer the rank of `x`. '
                         'Please set `tensor_rank` explicitly.')

    x_masked = mask_sequences(x, x_length, dtype=dtype, tensor_rank=tensor_rank)
    zeros_y = tf.zeros_like(y)
    x_aug = tf.concat([x_masked, zeros_y], axis=1)

    zeros_x = tf.zeros_like(x)
    y_aug = tf.concat([zeros_x, y], axis=1)

    # Now, x_aug.shape == y_aug.shape

    max_length_x = tf.shape(x)[1]
    batch_size = tf.shape(x)[0]

    initial_index = tf.constant(0, dtype=tf.int32)
    initial_outputs_ta = tf.TensorArray(
        dtype=dtype or x.dtype,
        size=0,
        dynamic_size=True)

    def _cond(index, _):
        return tf.less(index, batch_size)

    def _body(index, outputs_ta):
        y_aug_i_rolled = tf.roll(
            input=y_aug[index],
            shift=x_length[index] - max_length_x,  # shift to left
            axis=0)
        xy = x_aug[index] + y_aug_i_rolled
        return [index + 1, outputs_ta.write(index, xy)]

    res = tf.while_loop(_cond, _body, [initial_index, initial_outputs_ta])

    return res[1].stack()


def varlength_concat_py(x, y, x_length, dtype=None):
    """Concatenates rows of `x` and `y` where each row of
    `x` has a variable length.

    The function has the same semantic as :func:`varlength_concat`,
    except that this function is for numpy arrays instead of TF tensors.

    Both `x` and `y` are of numeric dtypes, such as `int32` and `float32`,
    with mask value `0`. The two arrays must be of the same dtype.

    Args:
        x: A array of shape `[batch_size, x_dim_2, other_dims]`.
        y: A array of shape `[batch_size, y_dim_2, other_dims]`.
            All dimensions except the 2nd dimension must be the same
            with those of `x`.
        x_length: A 1D int array of shape `[batch_size]` containing
            the length of each `x` row.
            Elements beyond the respective lengths will be
            made zero.
        dtype: Type of :attr:`x`. If `None`, inferred from
            :attr:`x` automatically.

    Returns:
        An array of shape `[batch_size, x_dim_2 + y_dim_2, other_dims]`.

    Example:
        .. code-block:: python

            x = np.asarray([[1, 1, 0, 0],
                            [1, 1, 1, 0]])
            x_length = [2, 3]
            y = np.asarray([[2, 2, 0],
                            [2, 2, 2]])

            out = varlength_concat_py(x, y, x_length)
            # out = [[1, 1, 2, 2, 0, 0, 0]
            #        [1, 1, 1, 2, 2, 2, 0]]
    """
    x = np.asarray(x, dtype=dtype)
    y = np.asarray(y, dtype=dtype)

    x_masked = mask_sequences(x, x_length, dtype=dtype)
    zeros_y = np.zeros_like(y)
    x_aug = np.concatenate([x_masked, zeros_y], axis=1)

    zeros_x = np.zeros_like(x)
    y_aug = np.concatenate([zeros_x, y], axis=1)

    # Now, x_aug.shape == y_aug.shape

    max_length_x = x.shape[1]
    batch_size = x.shape[0]

    for index in np.arange(batch_size):
        y_aug_i_rolled = np.roll(
            a=y_aug[index],
            shift=x_length[index] - max_length_x,
            axis=0)
        x_aug[index] += y_aug_i_rolled

    return x_aug


def varlength_roll(input, shift, axis=1, dtype=None):
    """Rolls the elements of *each row* of a tensor along an axis for
    variable steps.

    This is a `tf.while_loop` wrapper of :tf_main:`tf.roll <roll>`. Note the
    different definition of :attr:`shift` and :attr:`axis` here compared
    to :tf_main:`tf.roll <roll>`.

    Args:
        input: A tensor of shape `[batch_size, other_dims]` where
            `other_dims` can be multiple dimensions.
        shift: A 1D int tensor of shape `[batch_size]` containing
            the steps for which each row in the batch are rolled.
            Positive shifts will roll towards larger indices, while
            negative shifts will roll towards smaller indices.
        axis: A scalar int tensor > 0. The dimension that the roll
            should occur.
        dtype: Type of :attr:`input`. If `None`, inferred from
            :attr:`input` automatically.

    Returns:
        A Tensor of the same shape/dtype as :attr:`input`.

    Example:
        .. code-block:: python

            x = tf.constant([[0, 0, 1, 0],
                             [0, 1, 1, 1]])
            shift = [-2, -1]

            out = varlength_roll(x, shift)
            # out = [[1, 0, 0, 0]
            #        [1, 1, 1, 0]]


        .. code-block:: python

            x = tf.constant([[1, 2, 3, 4],
                             [5, 6, 7, 8]])
            shift = [1, -1]

            out = varlength_roll(x, shift)
            # out = [[4, 1, 2, 3]
            #        [6, 7, 8, 5]]
    """
    x = tf.convert_to_tensor(input)
    # x = input
    shift = tf.convert_to_tensor(shift)

    batch_size = tf.shape(x)[0]

    initial_index = tf.constant(0, dtype=tf.int32)
    initial_outputs_ta = tf.TensorArray(
        dtype=dtype or x.dtype,
        size=0,
        dynamic_size=True)

    def _cond(index, _):
        return tf.less(index, batch_size)

    def _body(index, outputs_ta):
        x_i_rolled = tf.roll(
            input=x[index],
            shift=shift[index],
            axis=axis - 1)
        return [index + 1, outputs_ta.write(index, x_i_rolled)]

    res = tf.while_loop(_cond, _body, [initial_index, initial_outputs_ta])

    return res[1].stack()
