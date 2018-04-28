#
"""
Various utilities for losses.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn          # pylint: disable=E0611

# pylint: disable=invalid-name, not-context-manager, protected-access,
# pylint: disable=too-many-arguments

__all__ = [
    "mask_and_reduce",
    "mask_sequences",
    "reduce_batch_time"
]

def mask_and_reduce(sequence,
                    sequence_length,
                    rank=2,
                    average_across_batch=True,
                    average_across_timesteps=False,
                    average_across_remaining=False,
                    sum_over_batch=False,
                    sum_over_timesteps=True,
                    sum_over_remaining=True,
                    dtype=None,
                    time_major=False):
    """Masks out sequence entries that are beyond the respective sequence
    lengths, and reduces (average or sum) away dimensions.

    This is a combined function of :func:`_mask_sequences` and
    :func:`_reduce_batch_time`.

    Args:
        sequence: A Tensor of sequence values.

            If `time_major=False` (default), this must be a Tensor of shape:
                `[batch_size, max_time, d_2, ..., d_rank]`, where the rank of
                the Tensor is specified with :attr:`rank`.

            If `time_major=True`, this must be a Tensor of shape:
                `[max_time, batch_size, d_2, ..., d_rank].`
        sequence_length: A Tensor of shape `[batch_size]`. Time steps beyond
            the respective sequence lengths will be made zero.
        average_across_timesteps (bool): If set, average the sequence across
            the time dimension. Must not set :attr:`average_across_timesteps`
            and :attr:`sum_over_timesteps` at the same time.
        average_across_batch (bool): If set, average the sequence across the
            batch dimension. Must not set :attr:`average_across_batch`'
            and :attr:`sum_over_batch` at the same time.
        average_across_remaining (bool): If set, average the sequence across the
            remaining dimensions. Must not set :attr:`average_across_remaining`'
            and :attr:`sum_over_remaining` at the same time.
        sum_over_timesteps (bool): If set, sum the loss across the
            time dimension. Must not set :attr:`average_across_timesteps`
            and :attr:`sum_over_timesteps` at the same time.
        sum_over_batch (bool): If set, sum the loss across the
            batch dimension. Must not set :attr:`average_across_batch`
            and :attr:`sum_over_batch` at the same time.
        sum_over_remaining (bool): If set, sum the loss across the
            remaining dimension. Must not set :attr:`average_across_remaining`
            and :attr:`sum_over_remaining` at the same time.
        rank (int): The rank of :attr:`sequence`. Default is 2, i.e.,
            :attr:`sequence` is a 2D Tensor consisting of batch and time
            dimensions.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`sequence` must have shape `[max_time, batch_size, ...]`.
            If `False` (default), :attr:`sequence` must have
            shape `[batch_size, max_time, ...]`.
        dtype (dtype): Type of :attr:`sequence`. If `None`, infer from
            :attr:`sequence` automatically.
    """
    if time_major:
        sequence = rnn._transpose_batch_time(sequence)

    sequence = mask_sequences(
        sequence, sequence_length, rank=rank, dtype=dtype, time_major=False)

    if rank > 2:
        if average_across_remaining and sum_over_remaining:
            raise ValueError("Only one of `average_across_remaining` and "
                             "`sum_over_remaining` can be set.")
        if average_across_remaining:
            sequence = tf.reduce_mean(sequence, axis=range(2, rank))
        elif sum_over_remaining:
            sequence = tf.reduce_sum(sequence, axis=range(2, rank))

    sequence = reduce_batch_time(sequence,
                                 sequence_length,
                                 average_across_batch,
                                 average_across_timesteps,
                                 sum_over_batch,
                                 sum_over_timesteps)

    reduce_time = average_across_timesteps or sum_over_timesteps
    reduce_batch = average_across_batch or sum_over_batch
    if not reduce_time and not reduce_batch and time_major:
        sequence = rnn._transpose_batch_time(sequence)

    return sequence


def mask_sequences(sequence,
                   sequence_length,
                   rank=2,
                   dtype=None,
                   time_major=False):
    """Masks out sequence entries that are beyond the respective sequence
    lengths.

    Args:
        sequence: A Tensor of sequence values.

            If `time_major=False` (default), this must be a Tensor of shape:
                `[batch_size, max_time, d_2, ..., d_rank]`, where the rank of
                the Tensor is specified with :attr:`rank`.

            If `time_major=True`, this must be a Tensor of shape:
                `[max_time, batch_size, d_2, ..., d_rank].`
        sequence_length: A Tensor of shape `[batch_size]`. Time steps beyond
            the respective sequence lengths will be made zero.
        rank (int): The rank of :attr:`sequence`. Default is 2, i.e.,
            :attr:`sequence` is a 2D Tensor consisting of batch and time
            dimensions.
        dtype (dtype): Type of :attr:`sequence`. If `None`, infer from
            :attr:`sequence` automatically.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`sequence` must have shape
            `[max_time, batch_size, d_2, ..., d_rank]`.
            If `False` (default), :attr:`sequence` must have
            shape `[batch_size, max_time, d_2, ..., d_rank]`.

    Returns:
        The masked sequence, i.e., a Tensor of the same shape as
        :attr:`sequence` but with masked-out entries (set to zero).
    """
    if rank < 2:
        raise ValueError("rank must be > 2. Got rank = {}".format(rank))
    if time_major:
        sequence = rnn._transpose_batch_time(sequence)
    max_time = tf.to_int32(tf.shape(sequence)[1])
    if dtype is None:
        dtype = sequence.dtype
    mask = tf.sequence_mask(
        tf.to_int32(sequence_length), max_time, dtype=dtype)
    for _ in range(2, rank):
        mask = tf.expand_dims(mask, axis=-1)
    sequence = sequence * mask
    if time_major:
        sequence = rnn._transpose_batch_time(sequence)
    return sequence

def reduce_batch_time(sequence,
                      sequence_length,
                      average_across_batch=True,
                      average_across_timesteps=False,
                      sum_over_batch=False,
                      sum_over_timesteps=True):
    """Average or sum over the respective dimensions of :attr:`sequence`, which
    is of shape `[batch_size, max_time, ...]`.
    """
    if average_across_timesteps and sum_over_timesteps:
        raise ValueError("Only one of `average_across_timesteps` and "
                         "`sum_over_timesteps` can be set.")
    if average_across_batch and sum_over_batch:
        raise ValueError("Only one of `average_across_batch` and "
                         "`sum_over_batch` can be set.")
    reduce_time = average_across_timesteps or sum_over_timesteps
    reduce_batch = average_across_batch or sum_over_batch
    if reduce_time:
        sequence = tf.reduce_sum(sequence, axis=[1])
    if average_across_timesteps:
        sequence = sequence / tf.to_float(sequence_length)
    if reduce_batch:
        sequence = tf.reduce_sum(sequence, axis=[0])
    if average_across_batch:
        sequence = sequence / tf.to_float(tf.shape(sequence_length)[0])
    return sequence
