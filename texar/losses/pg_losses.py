#
"""
Various loss functions for policy gradients.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

__all__ = [
    "pg_loss_with_logits",
    "pg_loss_with_log_probs"
]

#TODO(zhiting): allow average over batch, etc
def pg_loss_with_logits(logits, actions, advantages):
    """Policy gradient loss with logits. Used for discrete actions.

    pg_loss = mean( SG(advantages) * -log_prob( SG(actions) )  ),
    where SG(.) is stop_gradient(.)

    Args:
        logits: Unscaled log probabilities of shape
            `[d_0, d_1, ..., d_{r-1}, num_action_types]` and dtype `float32` or
            `float64`.
        actions: Tensor of shape `[d_0, d_1, ..., d_{r-1}]` and dtype
            `int32` or `int64`.
        advantages: Tensor of shape `[d_0, d_1, ..., d_{r-1}]`.

    Returns:
        A scalar Tensor of the loss to minimize.
    """
    actions = tf.stop_gradient(actions)
    advantages = tf.stop_gradient(advantages)
    neg_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=actions)
    return tf.reduce_mean(neg_log_probs * advantages)

def pg_loss_with_log_probs(log_probs, advantages):
    """Policy gradient loss with log probs of actions.

    pg_loss = mean( SG(advantages) * -log_probs ),
    where SG(.) is stop_gradient(.)

    Args:
        logits: Unscaled log probabilities of shape
            `[d_0, d_1, ..., d_{r-1}, num_action_types]` and dtype `float32` or
            `float64`.
        actions: Tensor of shape `[d_0, d_1, ..., d_{r-1}]` and dtype
            `int32` or `int64`.
        advantages: Tensor of shape `[d_0, d_1, ..., d_{r-1}]`.

    Returns:
        A scalar Tensor of the loss to minimize.

    """
    advantages = tf.stop_gradient(advantages)
    return tf.reduce_mean(-log_probs * advantages)
