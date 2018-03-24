#
"""
Various losses
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn          # pylint: disable=E0611

# pylint: disable=invalid-name, not-context-manager, protected-access

#TODO(zhiting): update the docs
# allow dtype
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
        sequence = rnn._transpose_batch_time(sequence)
    max_time = tf.to_int32(tf.shape(sequence)[1])
    mask = tf.sequence_mask(
        tf.to_int32(sequence_length), max_time, tf.float32)
    sequence = sequence * mask
    if time_major:
        sequence = rnn._transpose_batch_time(sequence)
    return sequence


def sequence_softmax_cross_entropy(labels,
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
    with tf.name_scope(name, "sequence_softmax_cross_entropy"):
        losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        return _mask_sequences(losses, sequence_length, time_major)


def average_sequence_softmax_cross_entropy(labels,
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
    with tf.name_scope(name, "average_sequence_softmax_cross_entropy"):
        losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        losses = _mask_sequences(losses, sequence_length, time_major)
        mean_loss = tf.reduce_sum(losses) / tf.reduce_sum(tf.to_float(sequence_length))
        return mean_loss


def sequence_sparse_softmax_cross_entropy(labels,
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
    with tf.name_scope(name, "sequence_sparse_softmax_cross_entropy"):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        return _mask_sequences(losses, sequence_length, time_major)

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
    with tf.name_scope(name, "average_sequence_sparse_softmax_cross_entropy"):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        losses = _mask_sequences(losses, sequence_length, time_major)
        loss = tf.reduce_sum(losses) / tf.to_float(tf.shape(labels)[0])
        return loss

def label_smoothing(labels, total_class, smooth_rate, name=None):
    with tf.name_scope(name, 'label_smoothing'):
        one_hot_labels = tf.one_hot(labels, depth=total_class)
        return  (1-smooth_rate)*one_hot_labels + (smooth_rate)/total_class

def smoothing_cross_entropy(logits,
                            labels,
                            vocab_size,
                            confidence,
                            gaussian=False):
  """Cross entropy with label smoothing to limit over-confidence.
  Args:
    logits: Tensor of size [batch_size, ?, ?, ?, vocab_size]
    labels: Tensor of size [batch_size, ?, ?, ?]
    vocab_size: Tensor representing the size of the vocabulary.
    confidence: Used to determine on and off values for label smoothing.
      If `gaussian` is true, `confidence` is the variance to the gaussian
      distribution.
    gaussian: Uses a gaussian distribution for label smoothing
  Returns:
  """
  with tf.name_scope("smoothing_cross_entropy", values=[logits, labels]):
    # Low confidence is given to all non-true labels, uniformly.
    low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)
    # Normalizing constant is the best cross-entropy value with soft targets.
    # We subtract it just for readability, makes no difference on learning.
    normalizing = -(
        confidence * tf.log(confidence) + tf.to_float(vocab_size - 1) *
        low_confidence * tf.log(low_confidence + 1e-20))

    if gaussian and confidence > 0.0:
      labels = tf.cast(labels, tf.float32)

      normal_dist = tf.distributions.Normal(loc=labels, scale=confidence)
      # Locations to evaluate the probability distributions.
      soft_targets = normal_dist.prob(
          tf.cast(tf.range(vocab_size), tf.float32)[:, None, None, None, None])
      # Reordering soft_targets from [vocab_size, batch_size, ?, ?, ?] to match
      # logits: [batch_size, ?, ?, ?, vocab_size]
      soft_targets = tf.transpose(soft_targets, perm=[1, 2, 3, 4, 0])
    else:
      soft_targets = tf.one_hot(
          tf.cast(labels, tf.int32),
          depth=vocab_size,
          on_value=confidence,
          off_value=low_confidence)
    xentropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=soft_targets)
    return xentropy - normalizing

