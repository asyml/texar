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


def reinforce_loss(sample_fn, reward_global, reward_local=None, num_samples=1):
    """Compute reinforce loss with global and local reward functions.

    Args:
        sample_fn: A callable function which takes an int as the number of samples and returns the tuple of
                    sampled sequences, corresponding probabilities and sequence lengths.
        reward_global: A callable function which takes sequences and lengths as arguments and
                    returns the rewards of the whole sequences.
        reward_local: None or a callable function which takes sequences and lengths as arguments and
                    returns the local rewards at each time step of the sequences.
        num_samples: the number of sequences to sample.

    Returns: A scalar of reinforce loss.

    """

    # shape = [batch, length]
    sequences, probs, seq_lens = sample_fn(num_samples)
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
    return -tf.reduce_mean(tf.reduce_sum(log_probs * rewards, axis=1) / seq_lens)


def reinforce_loss_with_MCtree(sample_fn, reward_global, reward_local=None, num_samples=1):
    """Compute reinforce loss with MC tree.

    Args:
        sample_fn: A callable function which takes an int as the number of samples and returns the tuple of
                    sampled sequences, corresponding probabilities and sequence lengths.
        reward_global: A callable function which takes sequences and lengths as arguments and
                    returns the rewards of the whole sequences.
        reward_local: None or a callable function which takes sequences and lengths as arguments and
                    returns the local rewards at each time step of the sequences.
        num_samples: the number of sequences to sample.

    Returns: A scalar of reinforce loss.

    """
    pass
