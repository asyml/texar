# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Modifications copyright (C) 2019 Texar
# ==============================================================================
"""
Utility functions for decoding. This file is modified from
`tf.contrib.seq2seq.dynamic_decode`.
"""

# pylint: disable=invalid-name, no-member, protected-access

import tensorflow as tf
from tensorflow.contrib.seq2seq import Decoder as TFDecoder
from tensorflow.python.framework import tensor_shape, tensor_util
from tensorflow.python.util import nest


__all__ = [
    "dynamic_decode"
]


def _concat(prefix, suffix, static=False):
    r"""Concat that enables int, Tensor, or TensorShape values.
    This function takes a size specification, which can be an integer, a
    TensorShape, or a Tensor, and converts it into a concatenated Tensor
    (if static = False) or a list of integers (if static = True).

    Args:
        prefix: The prefix; usually the batch size (and/or time step size).
          (TensorShape, int, or Tensor.)
        suffix: TensorShape, int, or Tensor.
        static: If `True`, return a python list with possibly unknown
          dimensions. Otherwise return a `Tensor`.

    Returns:
        shape: the concatenation of prefix and suffix.

    Raises:
        ValueError: if `suffix` is not a scalar or vector (or TensorShape).
        ValueError: if prefix or suffix was `None` and asked for dynamic
          Tensors out.
    """
    if isinstance(prefix, tf.Tensor):
        p = prefix
        p_static = tensor_util.constant_value(prefix)
        if p.shape.ndims == 0:
            p = tf.expand_dims(p, 0)
        elif p.shape.ndims != 1:
            raise ValueError("prefix tensor must be either a scalar or vector, "
                             "but saw tensor: %s" % p)
    else:
        p = tensor_shape.as_shape(prefix)
        p_static = p.as_list() if p.ndims is not None else None
        p = (
            tf.constant(p.as_list(), dtype=tf.int32)
            if p.is_fully_defined() else None)
    if isinstance(suffix, tf.Tensor):
        s = suffix
        s_static = tensor_util.constant_value(suffix)
        if s.shape.ndims == 0:
            s = tf.expand_dims(s, 0)
        elif s.shape.ndims != 1:
            raise ValueError("suffix tensor must be either a scalar or vector, "
                             "but saw tensor: %s" % s)
    else:
        s = tensor_shape.as_shape(suffix)
        s_static = s.as_list() if s.ndims is not None else None
        s = (
            tf.constant(s.as_list(), dtype=tf.int32)
            if s.is_fully_defined() else None)

    if static:
        shape = tensor_shape.as_shape(p_static).concatenate(s_static)
        shape = shape.as_list() if shape.ndims is not None else None
    else:
        if p is None or s is None:
            raise ValueError("Provided a prefix or suffix of None: %s and %s" %
                             (prefix, suffix))
        shape = tf.concat((p, s), 0)
    return shape


def _zero_state_tensors(state_size, batch_size, dtype):
    r"""Create tensors of zeros based on state_size, batch_size, and dtype."""

    def get_state_shape(s):
        r"""Combine s with batch_size to get a proper tensor shape."""

        c = _concat(batch_size, s)
        size = tf.zeros(c, dtype=dtype)
        return size

    return nest.map_structure(get_state_shape, state_size)


def _create_zero_outputs(size, dtype, batch_size):
    r"""Create a zero outputs Tensor structure."""

    def _create(s, d):
        return _zero_state_tensors(s, batch_size, d)

    return nest.map_structure(_create, size, dtype)


def _transpose_batch_time(x):
    r"""Transposes the batch and time dimensions of a Tensor.

    If the input tensor has rank < 2 it returns the original tensor. Retains as
    much of the static shape information as possible.

    Args:
        x: A Tensor.

    Returns:
        x transposed along the first two dimensions.
    """
    x_static_shape = x.get_shape()
    if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
        return x

    x_rank = tf.rank(x)
    x_t = tf.transpose(
        x, tf.concat(([1, 0], tf.range(2, x_rank)), axis=0))
    x_t.set_shape(
        tensor_shape.TensorShape(
            [x_static_shape.dims[1].value,
             x_static_shape.dims[0].value]).concatenate(x_static_shape[2:]))
    return x_t


def dynamic_decode(decoder,
                   output_time_major=False,
                   impute_finished=False,
                   maximum_iterations=None,
                   parallel_iterations=32,
                   swap_memory=False,
                   scope=None):
    r"""Perform dynamic decoding with `decoder`.

    Calls initialize() once and step() repeatedly on the Decoder object.

    Args:
      decoder: A `Decoder` instance.
      output_time_major: Python boolean.  Default: `False` (batch major).  If
        `True`, outputs are returned as time major tensors (this mode is
        faster). Otherwise, outputs are returned as batch major tensors
        (this adds extra time to the computation).
      impute_finished: Python boolean.  If `True`, then states for batch
        entries which are marked as finished get copied through and the
        corresponding outputs get zeroed out.  This causes some slowdown at
        each time step, but ensures that the final state and outputs have
        the correct values and that backprop ignores time steps that were
        marked as finished.
      maximum_iterations: `int32` scalar, maximum allowed number of decoding
        steps.  Default is `None` (decode until the decoder is fully done).
      parallel_iterations: Argument passed to `tf.while_loop`.
      swap_memory: Argument passed to `tf.while_loop`.
      scope: Optional variable scope to use.

    Returns:
      `(final_outputs, final_state, final_sequence_lengths)`.
    Raises:
      TypeError: if `decoder` is not an instance of `Decoder`.
      ValueError: if `maximum_iterations` is provided but is not a scalar.
    """
    if not isinstance(decoder, TFDecoder):
        raise TypeError("Expected decoder to be type Decoder, but saw: %s" %
                        type(decoder))

    with tf.variable_scope(scope, "decoder") as varscope:
        # Properly cache variable values inside the while_loop
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        if maximum_iterations is not None:
            maximum_iterations = tf.convert_to_tensor(
                maximum_iterations, dtype=tf.int32, name="maximum_iterations")
            if maximum_iterations.get_shape().ndims != 0:
                raise ValueError("maximum_iterations must be a scalar")

        initial_finished, initial_inputs, initial_state = decoder.initialize()

        zero_outputs = _create_zero_outputs(decoder.output_size,
                                            decoder.output_dtype,
                                            decoder.batch_size)

        if maximum_iterations is not None:
            initial_finished = tf.logical_or(
                initial_finished, 0 >= maximum_iterations)
        initial_sequence_lengths = tf.zeros_like(
            initial_finished, dtype=tf.int32)
        initial_time = tf.constant(0, dtype=tf.int32)

        def _shape(batch_size, from_shape):
            if (not isinstance(from_shape, tensor_shape.TensorShape) or
                    from_shape.ndims == 0):
                return None
            else:
                batch_size = tensor_util.constant_value(
                    tf.convert_to_tensor(
                        batch_size, name="batch_size"))
                return tensor_shape.TensorShape([batch_size]).\
                    concatenate(from_shape)

        dynamic_size = True

        def _create_ta(s, d):
            return tf.TensorArray(
                dtype=d,
                size=0 if dynamic_size else maximum_iterations,
                dynamic_size=dynamic_size,
                element_shape=_shape(decoder.batch_size, s))

        initial_outputs_ta = nest.map_structure(_create_ta, decoder.output_size,
                                                decoder.output_dtype)

        def condition(unused_time, unused_outputs_ta, unused_state,
                      unused_inputs, finished, unused_sequence_lengths):
            cond = tf.logical_not(tf.reduce_all(finished))
            cond_time = (maximum_iterations is None or
                         unused_time < maximum_iterations)
            ret = tf.logical_and(cond, tf.convert_to_tensor(cond_time))
            return ret

        def body(time, outputs_ta, state, inputs, finished, sequence_lengths):
            r"""Internal while_loop body.

            Args:
                time: scalar int32 tensor.
                outputs_ta: structure of TensorArray.
                state: (structure of) state tensors and TensorArrays.
                inputs: (structure of) input tensors.
                finished: bool tensor (keeping track of what's finished).
                sequence_lengths: int32 tensor (keeping track of time of
                    finish).

            Returns:
                `(time + 1, outputs_ta, next_state, next_inputs, next_finished,
                next_sequence_lengths)`.
            """
            (next_outputs, state) = decoder.step(time, inputs, state)

            # Check if the maximum iteration is met. If it is met, do not
            # compute the next inputs.
            reach_max = tf.equal(time + 1, maximum_iterations)
            (decoder_finished, next_inputs, decoder_state) = tf.cond(
                reach_max,
                lambda: (tf.cast(tf.ones_like(finished), tf.bool),
                         inputs, state),
                lambda: decoder.next_inputs(time, next_outputs, state)
            )
            if decoder.tracks_own_finished:
                next_finished = decoder_finished
            else:
                next_finished = tf.logical_or(decoder_finished, finished)
            next_sequence_lengths = tf.where(
                tf.logical_not(finished),
                tf.fill(tf.shape(sequence_lengths), time + 1),
                sequence_lengths)

            nest.assert_same_structure(state, decoder_state)
            nest.assert_same_structure(outputs_ta, next_outputs)
            nest.assert_same_structure(inputs, next_inputs)

            # Zero out output values past finish
            if impute_finished:
                emit = nest.map_structure(
                    lambda out, zero: tf.where(finished, zero, out),
                    next_outputs,
                    zero_outputs)
            else:
                emit = next_outputs

            # Copy through states past finish
            def _maybe_copy_state(new, cur):
                # TensorArrays and scalar states get passed through.
                if isinstance(cur, tf.TensorArray):
                    pass_through = True
                else:
                    new.set_shape(cur.shape)
                    pass_through = (new.shape.ndims == 0)
                return new if pass_through else tf.where(finished, cur, new)

            if impute_finished:
                next_state = nest.map_structure(
                    _maybe_copy_state, decoder_state, state)
            else:
                next_state = decoder_state

            outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
                                            outputs_ta, emit)
            return (time + 1, outputs_ta, next_state, next_inputs,
                    next_finished, next_sequence_lengths)

        res = tf.while_loop(
            condition,
            body,
            loop_vars=(
                initial_time,
                initial_outputs_ta,
                initial_state,
                initial_inputs,
                initial_finished,
                initial_sequence_lengths,
            ),
            parallel_iterations=parallel_iterations,
            maximum_iterations=maximum_iterations,
            swap_memory=swap_memory)

        final_outputs_ta = res[1]
        final_state = res[2]
        final_sequence_lengths = res[5]

        final_outputs = nest.map_structure(lambda ta: ta.stack(),
                                           final_outputs_ta)

        try:
            final_outputs, final_state = decoder.finalize(
                final_outputs, final_state, final_sequence_lengths)
        except NotImplementedError:
            pass

        if not output_time_major:
            final_outputs = nest.map_structure(_transpose_batch_time,
                                               final_outputs)

    return final_outputs, final_state, final_sequence_lengths
