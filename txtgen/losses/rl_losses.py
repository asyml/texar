#
"""
Various RL losses
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn  # pylint: disable=E0611

from txtgen.losses.mle_losses import _mask_sequences


def reinforce_loss(sample_fn,
                   reward_global_fn,
                   reward_local_fn=None,
                   num_samples=1):
    """Compute REINFORCE loss with global and local rewards.

    Args:
        sample_fn: A callable that takes :attr:`num_samples` and returns
            `(samples, probabilities, sequence_lengths)`, where:

            `samples` is a Tensor of shape `[num_samples, max_sequence_length]`
            containing the generated samples;

            `probabilities` is a Tensor of shape
            `[num_samples, max_sequence_length]` containing the probabilities of
            generating each position of the samples. Probabilities beyond the
            respective sequence lengths are ignored.

            `sequence_lengths` is a Tensor of shape `[num_samples]` containing
            the length of each samples.
        reward_global_fn: A callable that takes `(samples, sequence_lengths)`
            and returns a Tensor of shape `[num_samples]` containing the reward
            of each of the samples.
        reward_local_fn (optional): A callable that takes
            `(samples, sequence_lengths)` and returns a Tensor of shape
            `[num_samples, max_sequence_length]` containing the local reward
            at each time step of samples.
        num_samples (int scalar Tensor): the number of sequences to sample.

    Returns:
        A scalar Tensor of the REINFORCE loss.
    """

    # shape = [batch, length]
    sequences, probs, seq_lens = sample_fn(num_samples)
    # TODO(zhiting): `batch` is just `num_samples` ?
    batch, length = tf.shape(sequences)
    local_rewards = tf.constant(0., dtype=probs.dtype, shape=probs.shape)
    if reward_local_fn is not None:
        local_rewards = reward_local_fn(sequences, seq_lens)

    # shape = [batch, ]
    global_rewards = reward_global_fn(sequences, seq_lens)
    # add broadcast to global_rewards to match the shape of local_rewards
    rewards = local_rewards + tf.reshape(global_rewards, [batch, 1])

    eps = 1e-12
    log_probs = _mask_sequences(tf.log(probs + eps), seq_lens)
    loss = -tf.reduce_mean(
        tf.reduce_sum(log_probs * rewards, axis=1) / seq_lens)
    return loss


def reinforce_loss_with_MCtree(sample_fn,
                               reward_global_fn,
                               reward_local_fn=None,
                               num_samples=1):
    """Compute REINFORCE loss with Monte Carlo tree search.

    Args:
        sample_fn: A callable that takes :attr:`num_samples` and returns
            `(samples, probabilities, sequence_lengths)`, where:

            `samples` is a Tensor of shape `[num_samples, max_sequence_length]`
            containing the generated samples;

            `probabilities` is a Tensor of shape
            `[num_samples, max_sequence_length]` containing the probabilities of
            generating each position of the samples. Probabilities beyond the
            respective sequence lengths are ignored.

            `sequence_lengths` is a Tensor of shape `[num_samples]` containing
            the length of each samples.
        reward_global_fn: A callable that takes `(samples, sequence_lengths)`
            and returns a Tensor of shape `[num_samples]` containing the reward
            of each of the samples.
        reward_local_fn (optional): A callable that takes
            `(samples, sequence_lengths)` and returns a Tensor of shape
            `[num_samples, max_sequence_length]` containing the local reward
            at each time step of samples.
        num_samples (int scalar Tensor): the number of sequences to sample.

    Returns:
        A scalar Tensor of the REINFORCE loss.
    """
    raise NotImplementedError
