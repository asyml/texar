#
"""
Various losses
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn          # pylint: disable=E0611


def _mask_sequences(sequence, sequence_length, time_major=False):
    """Masks out sequence entries that are beyond the respective sequence
    lengths.

    Args:
        sequence: A Tensor of sequence values.

            If `time_major=False` (default), this must be a Tensor of shape:
                `[batch_size, max_time, (...), num_classes]`.

            If `time_major=True`, this must be a Tensor of shape:
                `[max_time, batch_size, (...), num_classes].`
        sequence_length: A Tensor of shape `[batch_size]`. Time steps beyond the
            respective sequence lengths will be made zero.
        time_major: The shape format of the inputs. If True, `sequence` must
            have shape `[max_time, batch_size, ...]`. If false (default),
            `sequence` must have shape `[batch_size, max_time, ...]`.

    Returns:
        A Tensor of the same shape as `sequence` but with masked-out entries.
    """
    if time_major:
        sequence = rnn._transpose_batch_time(sequence) # pylint: disable=protected-access
    max_time = tf.to_int32(tf.shape(sequence)[1])
    mask = tf.sequence_mask(
        tf.to_int32(sequence_length), max_time, tf.float32)
    sequence = sequence * mask
    if time_major:
        sequence = rnn._transpose_batch_time(sequence) # pylint: disable=protected-access
    return sequence


def sequence_softmax_cross_entropy(labels, # pylint: disable=invalid-name
                                   logits,
                                   sequence_length,
                                   time_major=False,
                                   name=None):
    """Computes softmax cross entropy for each time step of sequence
    predictions.

    Args:
        labels: Target class distributions.

            If `time_major=False` (default), this must be a Tensor of shape:
                `[batch_size, max_time, (...), num_classes]`.

            If `time_major=True`, this must be a Tensor of shape:
                `[max_time, batch_size, (...), num_classes].`
        logits: Unscaled log probabilities. This must have the same shape as
            `labels`.
        sequence_length: A Tensor of shape `[batch_size]`. Time steps beyond the
            respective sequence lengths will have zero losses.
        time_major: The shape format of the inputs. If True, `labels` and
            `logits` must have shape `[max_time, batch_size, ...]`. If false
            (default), `labels` and `logits` must have shape
            `[batch_size, max_time, ...]`.
        name: (optional) A name for the operation.

    Returns:
        A Tensor containing the loss for each time step of each example. Time
        steps beyond the respective sequence lengths will have zero losses.

        If `time_major=False` (default), this is of shape:
        `[batch_size, max_time, (...)]`.

        If `time_major=True`, this is of shape: `[max_time, batch_size, (...)]`.
    """
    # pylint: disable=not-context-manager
    with tf.name_scope(name, "sequence_softmax_cross_entropy"):
        losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        return _mask_sequences(losses, sequence_length, time_major)


def average_sequence_softmax_cross_entropy(labels, # pylint: disable=invalid-name
                                           logits,
                                           sequence_length,
                                           time_major=False,
                                           name=None):
    """Computes a single softmax cross entropy loss that averages over all time
    steps and all examples in a batch.

    See `sequence_softmax_cross_entropy` for the definition of arguments.

    Returns:
        A single average loss.
    """
    # pylint: disable=not-context-manager
    with tf.name_scope(name, "average_sequence_softmax_cross_entropy"):
        losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        losses = _mask_sequences(losses, sequence_length, time_major)
        seq_length_sum = tf.to_float(tf.reduce_sum(sequence_length))
        loss = tf.reduce_sum(losses) / seq_length_sum
        return loss


def sequence_sparse_softmax_cross_entropy(labels, # pylint: disable=invalid-name
                                          logits,
                                          sequence_length,
                                          time_major=False,
                                          name=None):
    """Computes sparse softmax cross entropy for each time step of sequence
    predictions.

    Args:
        labels: Target class indexes. I.e., classes are mutually exclusive (each
            entry is in exactly one class).

            If `time_major=False` (default), this must be a Tensor of shape:
                `[batch_size, max_time, (...)]`.

            If `time_major=True`, this must be a Tensor of shape:
                `[max_time, batch_size, (...)].`
        logits: Unscaled log probabilities. This must have the shape of
            `[max_time, batch_size, (...), num_classes]` or
            `[batch_size, max_time, (...), num_classes]` according to
            the value of `time_major`.
        sequence_length: A Tensor of shape `[batch_size]`. Time steps beyond the
            respective sequence lengths will have zero losses.
        time_major: The shape format of the inputs. If True, `labels` and
            `logits` must have shape `[max_time, batch_size, ...]`. If false
            (default), `labels` and `logits` must have shape
            `[batch_size, max_time, ...]`.
        name: (optional) A name for the operation.

    Returns:
        A Tensor containing the loss for each time step of each example.

        If `time_major=False` (default), this is of shape:
        `[batch_size, max_time, (...)]`.

        If `time_major=True`, this is of shape: `[max_time, batch_size, (...)]`.
    """
    # pylint: disable=not-context-manager
    with tf.name_scope(name, "sequence_sparse_softmax_cross_entropy"):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        return _mask_sequences(losses, sequence_length, time_major)

# pylint: disable=invalid-name
def average_sequence_sparse_softmax_cross_entropy(labels,
                                                  logits,
                                                  sequence_length,
                                                  time_major=False,
                                                  name=None):
    """Computes a single sparse softmax cross entropy loss that averages over
    all time steps and all examples in a batch.

    See `sequence_sparse_softmax_cross_entropy` for the definition of arguments.

    Returns:
        A single average loss.
    """
    # pylint: disable=not-context-manager
    with tf.name_scope(name, "average_sequence_sparse_softmax_cross_entropy"):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        losses = _mask_sequences(losses, sequence_length, time_major)
        seq_length_sum = tf.to_float(tf.reduce_sum(sequence_length))
        loss = tf.reduce_sum(losses) / seq_length_sum
        return loss

