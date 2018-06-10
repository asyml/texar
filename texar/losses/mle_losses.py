#
"""
Various losses
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.losses.losses_utils import mask_and_reduce
from texar.utils import shapes

# pylint: disable=invalid-name, not-context-manager, protected-access,
# pylint: disable=too-many-arguments

__all__ = [
    "sequence_softmax_cross_entropy",
    "sequence_sparse_softmax_cross_entropy",
    "sequence_sigmoid_cross_entropy"
]

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
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`labels` and :attr:`logits` must have shape
            `[max_time, batch_size, ...]`. If `False`
            (default), they must have shape `[batch_size, max_time, ...]`.
        stop_gradient_to_label (bool): If set, gradient propagation to
            :attr:`labels` will be disabled.
        name (str, optional): A name for the operation.

    Returns:
        A Tensor containing the loss, of rank 0, 1, or 2 depending on the
        arguments :attr:`{average_across}/{sum_over}_{timesteps}/{batch}`.
        For example:

        - If :attr:`sum_over_timesteps` and :attr:`average_across_batch`  \
        are `True` (default), the return Tensor is of rank 0.

        - If :attr:`average_across_batch` is `True` and other arguments are \
        `False`, the return Tensor is of shape `[max_time]`.
    """
    with tf.name_scope(name, "sequence_softmax_cross_entropy"):
        if stop_gradient_to_label:
            labels = tf.stop_gradient(labels)

        losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels, logits=logits)

        losses = mask_and_reduce(
            losses,
            sequence_length,
            rank=2,
            average_across_batch=average_across_batch,
            average_across_timesteps=average_across_timesteps,
            sum_over_batch=sum_over_batch,
            sum_over_timesteps=sum_over_timesteps,
            time_major=time_major)

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
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`labels` and :attr:`logits` must have shape
            `[max_time, batch_size, ...]`. If `False`
            (default), they must have shape `[batch_size, max_time, ...]`.
        name (str, optional): A name for the operation.

    Returns:
        A Tensor containing the loss, of rank 0, 1, or 2 depending on the
        arguments :attr:`{average_across}/{sum_over}_{timesteps}/{batch}`.
        For example:

        - If :attr:`sum_over_timesteps` and :attr:`average_across_batch`  \
        are `True` (default), the return Tensor is of rank 0.

        - If :attr:`average_across_batch` is `True` and other arguments are \
        `False`, the return Tensor is of shape `[max_time]`.
    """
    with tf.name_scope(name, "sequence_sparse_softmax_cross_entropy"):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)

        losses = mask_and_reduce(
            losses,
            sequence_length,
            rank=2,
            average_across_batch=average_across_batch,
            average_across_timesteps=average_across_timesteps,
            sum_over_batch=sum_over_batch,
            sum_over_timesteps=sum_over_timesteps,
            time_major=time_major)

        return losses

def sequence_sigmoid_cross_entropy(labels,
                                   logits,
                                   sequence_length,
                                   average_across_batch=True,
                                   average_across_timesteps=False,
                                   average_across_classes=True,
                                   sum_over_batch=False,
                                   sum_over_timesteps=True,
                                   sum_over_classes=False,
                                   time_major=False,
                                   stop_gradient_to_label=False,
                                   name=None):
    """Computes sigmoid cross entropy for each time step of sequence
    predictions.

    Args:
        labels: Target class distributions.

            If :attr:`time_major` is `False` (default), this must be a
                Tensor of shape `[batch_size, max_time(, num_classes)]`.

            If :attr:`time_major` is `True`, this must be a Tensor of shape
                `[max_time, batch_size(, num_classes)]`.

            Each row of :attr:`labels` should be a valid probability
            distribution, otherwise, the computation of the gradient will be
            incorrect.
        logits: Unscaled log probabilities having the same shape as with
            :attr:`labels`.
        sequence_length: A Tensor of shape `[batch_size]`. Time steps beyond
            the respective sequence lengths will have zero losses.
        average_across_timesteps (bool): If set, average the loss across
            the time dimension. Must not set :attr:`average_across_timesteps`
            and :attr:`sum_over_timesteps` at the same time.
        average_across_batch (bool): If set, average the loss across the
            batch dimension. Must not set :attr:`average_across_batch`'
            and :attr:`sum_over_batch` at the same time.
        average_across_classes (bool): If set, average the loss across the
            class dimension (if exists). Must not set
            :attr:`average_across_classes`' and :attr:`sum_over_classes` at
            the same time. Ignored if :attr:`logits` is a 2D Tensor.
        sum_over_timesteps (bool): If set, sum the loss across the
            time dimension. Must not set :attr:`average_across_timesteps`
            and :attr:`sum_over_timesteps` at the same time.
        sum_over_batch (bool): If set, sum the loss across the
            batch dimension. Must not set :attr:`average_across_batch`
            and :attr:`sum_over_batch` at the same time.
        sum_over_classes (bool): If set, sum the loss across the
            class dimension. Must not set :attr:`average_across_classes`
            and :attr:`sum_over_classes` at the same time. Ignored if
            :attr:`logits` is a 2D Tensor.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`labels` and :attr:`logits` must have shape
            `[max_time, batch_size, ...]`. If `False`
            (default), they must have shape `[batch_size, max_time, ...]`.
        stop_gradient_to_label (bool): If set, gradient propagation to
            :attr:`labels` will be disabled.
        name (str, optional): A name for the operation.

    Returns:
        A Tensor containing the loss, of rank 0, 1, or 2 depending on the
        arguments :attr:`{average_across}/{sum_over}_{timesteps}/{batch}`.
        For example:

        - If :attr:`sum_over_timesteps` and :attr:`average_across_batch`  \
        are `True` (default), the return Tensor is of rank 0.

        - If :attr:`average_across_batch` is `True` and other arguments are \
        `False`, the return Tensor is of shape `[max_time]`.
    """

    with tf.name_scope(name, "sequence_sigmoid_cross_entropy"):
        if stop_gradient_to_label:
            labels = tf.stop_gradient(labels)

        losses = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits)

        rank = shapes.get_rank(logits) or shapes.get_rank(labels)
        if rank is None:
            raise ValueError(
                'Cannot determine the rank of `logits` or `labels`.')

        losses = mask_and_reduce(
            losses,
            sequence_length,
            rank=rank,
            average_across_batch=average_across_batch,
            average_across_timesteps=average_across_timesteps,
            average_across_remaining=average_across_classes,
            sum_over_batch=sum_over_batch,
            sum_over_timesteps=sum_over_timesteps,
            sum_over_remaining=sum_over_classes,
            time_major=time_major)

        return losses
