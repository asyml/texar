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
Various RL losses
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.losses.mle_losses import _mask_sequences


def reinforce_loss(sample_fn,
                   global_reward_fn,
                   local_reward_fn=None,
                   num_samples=1):
    """Computes REINFORCE loss with global and local rewards.

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
        global_reward_fn: A callable that takes `(samples, sequence_lengths)`
            and returns a Tensor of shape `[num_samples]` containing the reward
            of each of the samples.
        local_reward_fn (optional): A callable that takes
            `(samples, sequence_lengths)` and returns a Tensor of shape
            `[num_samples, max_sequence_length]` containing the local reward
            at each time step of samples.
        num_samples (int scalar Tensor): the number of sequences to sample.

    Returns:
        A scalar Tensor of the REINFORCE loss.
    """

    # shape = [batch, length]
    sequences, probs, seq_lens = sample_fn(num_samples)
    batch, _ = tf.shape(sequences)
    rewards_local = tf.constant(0., dtype=probs.dtype, shape=probs.shape)
    if local_reward_fn is not None:
        rewards_local = local_reward_fn(sequences, seq_lens)

    # shape = [batch, ]
    rewards_global = global_reward_fn(sequences, seq_lens)
    # add broadcast to rewards_global to match the shape of rewards_local
    rewards = rewards_local + tf.reshape(rewards_global, [batch, 1])

    eps = 1e-12
    log_probs = _mask_sequences(tf.log(probs + eps), seq_lens)
    loss = - tf.reduce_mean(
        tf.reduce_sum(log_probs * rewards, axis=1) / seq_lens)
    return loss


def reinforce_loss_with_MCtree(sample_fn,   # pylint: disable=invalid-name
                               global_reward_fn,
                               local_reward_fn=None,
                               num_samples=1):
    """Computes REINFORCE loss with Monte Carlo tree search.

    Args:
        sample_fn: A callable that takes :attr:`num_samples`, 'given_actions'
            and returns `(samples, probabilities, sequence_lengths)`, where:

            `samples` is a Tensor of shape `[num_samples, max_sequence_length]`
            containing the generated samples;

            `probabilities` is a Tensor of shape
            `[num_samples, max_sequence_length]` containing the probabilities of
            generating each position of the samples. Probabilities beyond the
            respective sequence lengths are ignored.

            `sequence_lengths` is a Tensor of shape `[num_samples]` containing
            the length of each samples.
        global_reward_fn: A callable that takes `(samples, sequence_lengths)`
            and returns a Tensor of shape `[num_samples]` containing the reward
            of each of the samples.
        local_reward_fn (optional): A callable that takes
            `(samples, sequence_lengths)` and returns a Tensor of shape
            `[num_samples, max_sequence_length]` containing the local reward
            at each time step of samples.
        num_samples (int scalar Tensor): the number of sequences to sample.

    Returns:
        A scalar Tensor of the REINFORCE loss.
    """
    raise NotImplementedError
