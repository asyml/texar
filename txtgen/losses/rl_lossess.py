#
"""
Various RL losses
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn  # pylint: disable=E0611

from .mle_losses import _mask_sequences

def reinforce_loss(policy, reward_global, reward_local=None, num_samples=1):
    """Compute reinforce loss with global and local reward functions.

    Args:
        policy:
        reward_global:
        reward_local:
        num_samples:

    Returns:

    """

    # shape = [batch, length]
    sequences, probs, seq_lens = policy.sample(num_samples)
    batch, length = tf.shape(sequences)
    local_rewards = tf.constant(0., dtype=probs.dtype, shape=probs.shape)
    if reward_local is not None:
        local_rewards = reward_local(sequences, seq_lens)

    # shape = [batch, ]
    global_rewards = reward_global(sequences, seq_lens)
    # add broadcast to global_rewards to match the shape of local_rewards
    rewards = local_rewards + tf.reshape(global_rewards, [batch, 1])

    eps = 1e-12
    log_probs = _mask_sequences(tf.log(probs + eps), seq_lens)
    return -tf.reduce_mean(log_probs * rewards)


def reinforce_loss_with_MCtree(policy, reward_global, reward_local=None, num_samples=1):
    """Compute reinforce loss with MC tree.

    Args:
        policy:
        reward_global:
        reward_local:
        num_samples:

    Returns:

    """
    pass