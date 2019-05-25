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
Various loss functions for policy gradients.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.losses.losses_utils import mask_and_reduce
from texar.utils.shapes import get_rank

# pylint: disable=too-many-arguments, protected-access

__all__ = [
    "pg_loss_with_logits",
    "pg_loss_with_log_probs"
]

def pg_loss_with_logits(actions,
                        logits,
                        advantages,
                        rank=None,
                        batched=False,
                        sequence_length=None,
                        average_across_batch=True,
                        average_across_timesteps=False,
                        average_across_remaining=False,
                        sum_over_batch=False,
                        sum_over_timesteps=True,
                        sum_over_remaining=True,
                        time_major=False):
    """Policy gradient loss with logits. Used for discrete actions.

    `pg_loss = reduce( advantages * -log_prob( actions )  )`,
    where `advantages` and `actions` do not back-propagate gradients.

    All arguments except :attr:`logits` and :attr:`actions` are the same with
    :func:`pg_loss_with_log_probs`.

    Args:
        actions: Tensor of shape
            `[(batch_size,) max_time, d_3, ..., d_rank]` and of dtype
            `int32` or `int64`.
            The rank of the Tensor is specified with :attr:`rank`.

            The batch dimension exists only if :attr:`batched` is `True`.

            The batch and time dimensions
            are exchanged, i.e., `[max_time, batch_size, ...]` if
            :attr:`time_major` is `True`.
        logits: Unscaled log probabilities of shape
            `[(batch_size,) max_time, d_3, ..., d_{rank+1}]`
            and dtype `float32` or `float64`.
            The batch and time dimensions are exchanged if `time_major`
            is `True`.
        advantages: Tensor of shape
            `[(batch_size,) max_time, d_3, ..., d_rank]` and
            dtype `float32` or `float64`.
            The batch and time dimensions are exchanged if `time_major`
            is `True`.
        rank (int, optional): The rank of :attr:`actions`.
            If `None` (default), rank is automatically inferred from
            `actions` or `advantages`. If the inference fails,
            `rank` is set to 1 if :attr:`batched` is `False`,
            and set to 2 if :attr:`batched` is `True`.
        batched (bool): `True` if the inputs are batched.
        sequence_length (optional): A Tensor of shape `[batch_size]`.
            Time steps beyond the respective sequence lengths will have zero
            losses. Used if :attr:`batched` is `True`.
        average_across_timesteps (bool): If set, average the loss across
            the time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
        average_across_batch (bool): If set, average the loss across the
            batch dimension. Must not set `average_across_batch`'
            and `sum_over_batch` at the same time.
            Ignored if `batched` is `False`.
        average_across_remaining (bool): If set, average the sequence across the
            remaining dimensions. Must not set `average_across_remaining`'
            and `sum_over_remaining` at the same time. Ignored if
            no more dimensions other than the batch and time dimensions.
        sum_over_timesteps (bool): If set, sum the loss across the
            time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
        sum_over_batch (bool): If set, sum the loss across the
            batch dimension. Must not set `average_across_batch`
            and `sum_over_batch` at the same time.
            Ignored if `batched` is `False`.
        sum_over_remaining (bool): If set, sum the loss across the
            remaining dimension. Must not set `average_across_remaining`
            and `sum_over_remaining` at the same time. Ignored if
            no more dimensions other than the batch and time dimensions.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`logits`, :attr:`actions` and :attr:`advantages` must
            have shape `[max_time, batch_size, ...]`. If `False` (default),
            they must have shape `[batch_size, max_time, ...]`.
            Ignored if `batched` is `False`.

    Returns:
        A Tensor containing the loss to minimize, whose rank depends on the
        reduce arguments. For example, the batch dimension is reduced if
        either :attr:`average_across_batch` or :attr:`sum_over_batch` is
        `True`, which decreases the rank of output tensor by 1.
    """
    actions = tf.stop_gradient(actions)
    neg_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=actions)
    return pg_loss_with_log_probs(
        log_probs=-neg_log_probs,
        advantages=advantages,
        rank=rank,
        batched=batched,
        sequence_length=sequence_length,
        average_across_batch=average_across_batch,
        average_across_timesteps=average_across_timesteps,
        average_across_remaining=average_across_remaining,
        sum_over_batch=sum_over_batch,
        sum_over_timesteps=sum_over_timesteps,
        sum_over_remaining=sum_over_remaining,
        time_major=time_major)

def pg_loss_with_log_probs(log_probs,
                           advantages,
                           rank=None,
                           batched=False,
                           sequence_length=None,
                           average_across_batch=True,
                           average_across_timesteps=False,
                           average_across_remaining=False,
                           sum_over_batch=False,
                           sum_over_timesteps=True,
                           sum_over_remaining=True,
                           time_major=False):
    """Policy gradient loss with log probs of actions.

    `pg_loss = reduce( advantages * -log_probs )`,
    where `advantages` does not back-propagate gradients.

    All arguments except :attr:`log_probs` are the same as
    :func:`pg_loss_with_logits`.

    Args:
        log_probs: Log probabilities of shape
            `[(batch_size,) max_time, ..., d_rank]` and dtype `float32`
            or `float64`. The rank of the Tensor is specified
            with :attr:`rank`.

            The batch dimension exists only if :attr:`batched` is `True`.

            The batch and time dimensions are exchanged, i.e.,
            `[max_time, batch_size, ...]` if :attr:`time_major` is `True`.
        advantages: Tensor of shape
            `[(batch_size,) max_time, d_3, ..., d_rank]` and
            dtype `float32` or `float64`.
            The batch dimension exists only if `batched` is `True`.
            The batch and time dimensions
            are exchanged if `time_major` is `True`.
        rank (int, optional): The rank of :attr:`log_probs`.
            If `None` (default), rank is automatically inferred from
            `log_probs` or `advantages`. If the inference fails,
            `rank` is set to 1 if `batched``==False`,
            and set to 2 if `batched``==True`.
        batched (bool): `True` if the inputs are batched.
        sequence_length (optional): A Tensor of shape `[batch_size]`.
            Time steps beyond the respective sequence lengths will have zero
            losses. Used if :attr:`batched` is `True`.
        average_across_timesteps (bool): If set, average the loss across
            the time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
        average_across_batch (bool): If set, average the loss across the
            batch dimension. Must not set `average_across_batch`'
            and `sum_over_batch` at the same time.
            Ignored if `batched` is `False`.
        average_across_remaining (bool): If set, average the sequence across the
            remaining dimensions. Must not set `average_across_remaining`'
            and `sum_over_remaining` at the same time. Ignored if
            no more dimensions other than the batch and time dimensions.
        sum_over_timesteps (bool): If set, sum the loss across the
            time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
        sum_over_batch (bool): If set, sum the loss across the
            batch dimension. Must not set `average_across_batch`
            and `sum_over_batch` at the same time.
            Ignored if `batched` is `False`.
        sum_over_remaining (bool): If set, sum the loss across the
            remaining dimension. Must not set `average_across_remaining`
            and `sum_over_remaining` at the same time. Ignored if
            no more dimensions other than the batch and time dimensions.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`log_probs` and :attr:`advantages` must have shape
            `[max_time, batch_size, ...]`. If `False` (default),
            they must have shape `[batch_size, max_time, ...]`.
            Ignored if :attr:`batched` is `False`.

    Returns:
        A Tensor containing the loss to minimize, whose rank depends on the
        reduce arguments. For example, the batch dimension is reduced if
        either :attr:`average_across_batch` or :attr:`sum_over_batch` is
        `True`, which decreases the rank of output tensor by 1.
    """
    advantages = tf.stop_gradient(advantages)

    losses = -log_probs * advantages

    if rank is None:
        rank = get_rank(log_probs) or get_rank(advantages)
    if rank is None:
        rank = 2 if batched else 1

    if batched:
        losses = mask_and_reduce(
            losses,
            sequence_length,
            rank=rank,
            average_across_batch=average_across_batch,
            average_across_timesteps=average_across_timesteps,
            average_across_remaining=average_across_remaining,
            sum_over_batch=sum_over_batch,
            sum_over_timesteps=sum_over_timesteps,
            sum_over_remaining=sum_over_remaining,
            time_major=time_major)
    elif rank > 1:
        if average_across_remaining and sum_over_remaining:
            raise ValueError("Only one of `average_across_remaining` and "
                             "`sum_over_remaining` can be set.")
        if average_across_remaining:
            losses = tf.reduce_mean(losses, axis=list(range(1, rank)))
        elif sum_over_remaining:
            losses = tf.reduce_sum(losses, axis=list(range(1, rank)))

    if not batched:
        if average_across_timesteps and sum_over_timesteps:
            raise ValueError("Only one of `average_across_timesteps` and "
                             "`sum_over_timesteps` can be set.")
        if average_across_timesteps:
            losses = tf.reduce_mean(losses, axis=0)
        elif sum_over_timesteps:
            losses = tf.reduce_sum(losses, axis=0)

    return losses
