#
"""
Various losses
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn          # pylint: disable=E0611

# pylint: disable=invalid-name, not-context-manager, protected-access,
# pylint: disable=too-many-arguments

__all__ = [
    "sequence_softmax_cross_entropy",
    "sequence_sparse_softmax_cross_entropy"
]

def _mask_sequences(sequence, sequence_length, dtype=None,
                    time_major=False):
    """Masks out sequence entries that are beyond the respective sequence
    lengths.

    Args:
        sequence: A Tensor of sequence values.

            If `time_major=False` (default), this must be a Tensor of shape:
                `[batch_size, max_time, (...), num_classes]`.

            If `time_major=True`, this must be a Tensor of shape:
                `[max_time, batch_size, (...), num_classes].`
        sequence_length: A Tensor of shape `[batch_size]`. Time steps beyond
            the respective sequence lengths will be made zero.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`sequence` must have shape `[max_time, batch_size, ...]`.
            If `False` (default), :attr:`sequence` must have
            shape `[batch_size, max_time, ...]`.
        dtype (dtype): Type of :attr:`sequence`. If `None`, infer from
            :attr:`sequence` automatically.

    Returns:
        The masked sequence, i.e., a Tensor of the same shape as
        :attr:`sequence` but with masked-out entries (set to zero).
    """
    if time_major:
        sequence = rnn._transpose_batch_time(sequence)
    max_time = tf.to_int32(tf.shape(sequence)[1])
    if dtype is None:
        dtype = sequence.dtype
    mask = tf.sequence_mask(
        tf.to_int32(sequence_length), max_time, dtype=dtype)
    sequence = sequence * mask
    if time_major:
        sequence = rnn._transpose_batch_time(sequence)
    return sequence


def _reduce_batch_time(tensor,
                       sequence_length,
                       average_across_batch=True,
                       average_across_timesteps=False,
                       sum_over_batch=False,
                       sum_over_timesteps=True):
    """Average or sum over the respective dimensions of :attr:`tensor`, which
    is shape `[batch_size, max_dim]`.
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
        tensor = tf.reduce_sum(tensor, axis=[1])
    if average_across_timesteps:
        tensor = tensor / tf.to_float(sequence_length)
    if reduce_batch:
        tensor = tf.reduce_sum(tensor, axis=[0])
    if average_across_batch:
        tensor = tensor / tf.to_float(tf.shape(sequence_length)[0])
    return tensor

def sequence_softmax_cross_entropy(labels,
                                   logits,
                                   sequence_length,
                                   average_across_batch=True,
                                   average_across_timesteps=False,
                                   sum_over_batch=False,
                                   sum_over_timesteps=True,
                                   time_major=False,
                                   stop_gradient_to_label=False,
                                   name=None):
    """Computes softmax cross entropy for each time step of sequence
    predictions.

    Args:
        labels: Target class distributions.

            If :attr:`time_major` is `False` (default), this must be a
                Tensor of shape `[batch_size, max_time, num_classes]`.

            If :attr:`time_major` is `True`, this must be a Tensor of shape
                `[max_time, batch_size, num_classes]`.

            Each row of :attr:`labels` should be a valid probability
            distribution, otherwise, the computation of the gradient will be
            incorrect.
        logits: Unscaled log probabilities. This must have the shape of
            `[max_time, batch_size, num_classes]` or
            `[batch_size, max_time, num_classes]` according to
            the value of :attr:`time_major`.
        sequence_length: A Tensor of shape `[batch_size]`. Time steps beyond
            the respective sequence lengths will have zero losses.
        average_across_timesteps (bool): If set, average the loss across
            the time dimension. Must not set :attr:`average_across_timesteps`
            and :attr:`sum_over_timesteps` at the same time.
        average_across_batch (bool): If set, average the loss across the
            batch dimension. Must not set :attr:`average_across_batch`'
            and :attr:`sum_over_batch` at the same time.
        sum_over_timesteps (bool): If set, sum the loss across the
            time dimension. Must not set :attr:`average_across_timesteps`
            and :attr:`sum_over_timesteps` at the same time.
        sum_over_batch (bool): If set, sum the loss across the
            batch dimension. Must not set :attr:`average_across_batch`
            and :attr:`sum_over_batch` at the same time.
        time_major (bool): The shape format of the inputs. If True, `labels` and
            `logits` must have shape `[max_time, batch_size, ...]`. If false
            (default), `labels` and `logits` must have shape
            `[batch_size, max_time, ...]`.
        stop_gradient_to_label (bool): If set, gradient propagation to
            :attr:`labels` will be disabled.
        name (str, optional): A name for the operation.

    Returns:
        A Tensor containing the loss, of rank 0, 1, or 2 depending on the
        arguments :attr:`average_across/sum_over_timesteps/batch`. E.g.,
        if :attr:`sum_over_timesteps` and :attr:`average_across_batch` are
        `True` (default), the return Tensor is of rank 0.
        If :attr:`average_across_batch` is `True` and other arguments are
        `False`, the return Tensor is of shape `[max_time]`.
    """
    with tf.name_scope(name, "sequence_softmax_cross_entropy"):
        if time_major:
            labels = rnn._transpose_batch_time(labels)
            logits = rnn._transpose_batch_time(logits)
        if stop_gradient_to_label:
            labels = tf.stop_gradient(labels)
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels, logits=logits)
        losses = _mask_sequences(losses, sequence_length, time_major=False)

        losses = _reduce_batch_time(
            losses, sequence_length,
            average_across_batch, average_across_timesteps,
            sum_over_batch, sum_over_timesteps)

        reduce_time = average_across_timesteps or sum_over_timesteps
        reduce_batch = average_across_batch or sum_over_batch
        if not reduce_time and not reduce_batch and time_major:
            losses = rnn._transpose_batch_time(losses)

        return losses

def sequence_sparse_softmax_cross_entropy(labels,
                                          logits,
                                          sequence_length,
                                          average_across_batch=True,
                                          average_across_timesteps=False,
                                          sum_over_batch=False,
                                          sum_over_timesteps=True,
                                          time_major=False,
                                          name=None):
    """Computes sparse softmax cross entropy for each time step of sequence
    predictions.

    Args:
        labels: Target class indexes. I.e., classes are mutually exclusive
            (each entry is in exactly one class).

            If :attr:`time_major` is `False` (default), this must be
                a Tensor of shape `[batch_size, max_time]`.

            If :attr:`time_major` is `True`, this must be a Tensor of shape
                `[max_time, batch_size].`
        logits: Unscaled log probabilities. This must have the shape of
            `[max_time, batch_size, num_classes]` or
            `[batch_size, max_time, num_classes]` according to
            the value of :attr:`time_major`.
        sequence_length: A Tensor of shape `[batch_size]`. Time steps beyond
            the respective sequence lengths will have zero losses.
        average_across_timesteps (bool): If set, average the loss across
            the time dimension. Must not set :attr:`average_across_timesteps`
            and :attr:`sum_over_timesteps` at the same time.
        average_across_batch (bool): If set, average the loss across the
            batch dimension. Must not set :attr:`average_across_batch`'
            and :attr:`sum_over_batch` at the same time.
        sum_over_timesteps (bool): If set, sum the loss across the
            time dimension. Must not set :attr:`average_across_timesteps`
            and :attr:`sum_over_timesteps` at the same time.
        sum_over_batch (bool): If set, sum the loss across the
            batch dimension. Must not set :attr:`average_across_batch`
            and :attr:`sum_over_batch` at the same time.
        time_major (bool): The shape format of the inputs. If True, `labels` and
            `logits` must have shape `[max_time, batch_size, ...]`. If false
            (default), `labels` and `logits` must have shape
            `[batch_size, max_time, ...]`.
        name (str, optional): A name for the operation.

    Returns:
        A Tensor containing the loss, of rank 0, 1, or 2 depending on the
        arguments :attr:`average_across/sum_over_timesteps/batch`. E.g.,
        if :attr:`sum_over_timesteps` and :attr:`average_across_batch` are
        `True` (default), the return Tensor is of rank 0.
        If :attr:`average_across_batch` is `True` and other arguments are
        `False`, the return Tensor is of shape `[max_time]`.
    """
    with tf.name_scope(name, "sequence_sparse_softmax_cross_entropy"):
        if time_major:
            labels = rnn._transpose_batch_time(labels)
            logits = rnn._transpose_batch_time(logits)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        losses = _mask_sequences(losses, sequence_length, time_major=False)

        losses = _reduce_batch_time(
            losses, sequence_length,
            average_across_batch, average_across_timesteps,
            sum_over_batch, sum_over_timesteps)

        reduce_time = average_across_timesteps or sum_over_timesteps
        reduce_batch = average_across_batch or sum_over_batch
        if not reduce_time and not reduce_batch and time_major:
            losses = rnn._transpose_batch_time(losses)

        return losses

#TODO(zhiting): add docs
def label_smoothing(labels, total_class, smooth_rate, name=None):
    """TODO
    """
    with tf.name_scope(name, 'label_smoothing'):
        one_hot_labels = tf.one_hot(labels, depth=total_class)
        return  (1-smooth_rate)*one_hot_labels + (smooth_rate)/total_class
