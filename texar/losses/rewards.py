#
"""
Various reward related functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

from texar.utils.shapes import mask_sequences

# pylint: disable=invalid-name, too-many-arguments, no-member

__all__ = [
    "discount_reward"
]

def discount_reward(reward,
                    sequence_length=None,
                    ndims=1,
                    discount=1.,
                    normalize=False,
                    dtype=None):
    """TODO
    """
    is_tensor = tf.contrib.framework.is_tensor
    if is_tensor(reward) or is_tensor(sequence_length):
        if ndims == 1:
            disc_reward = _discount_reward_tensor_1d(
                reward, sequence_length, discount, dtype)
        elif ndims == 2:
            disc_reward = _discount_reward_tensor_2d(
                reward, sequence_length, discount, dtype)

        if normalize:
            mu, var = tf.nn.moments(disc_reward, axes=[0, 1], keep_dims=True)
            disc_reward = (disc_reward - mu) / (tf.sqrt(var) + 1e-8)
    else:
        raise NotImplementedError

    return disc_reward

def _discount_reward_py_1d(reward, sequence_length, discount=1., dtype=None):
    if sequence_length is None:
        raise ValueError('sequence_length must not be `None` for 1D reward.')

    reward = np.array(reward)
    sequence_length = np.array(sequence_length)

    batch_size = reward.shape[0]
    max_seq_length = np.max(sequence_length)
    dtype = dtype or reward.dtype

    if discount == 1.:
        dmat = np.ones([batch_size, max_seq_length], dtype=dtype)
    else:
        steps = np.tile(np.arange(max_seq_length), [batch_size, 1])
        mask = np.asarray(steps < (sequence_length-1)[:, None], dtype=dtype)
        # Make each row = [discount, ..., discount, 1, ..., 1]
        dmat = mask * discount + (1 - mask)
        dmat = np.cumprod(dmat[:, ::-1], axis=1)[:, ::-1]

    disc_reward = dmat * reward[:, None]
    disc_reward = mask_sequences(disc_reward, sequence_length, dtype=dtype)
    #mask = np.asarray(steps < sequence_length[:, None], dtype=dtype)
    #disc_reward = mask * disc_reward

    return disc_reward

def _discount_reward_tensor_1d(reward, sequence_length,
                               discount=1., dtype=None):
    if sequence_length is None:
        raise ValueError('sequence_length must not be `None` for 1D reward.')

    batch_size = tf.shape(reward)[0]
    max_seq_length = tf.reduce_max(sequence_length)
    dtype = dtype or reward.dtype

    if discount == 1.:
        dmat = tf.ones(
            tf.concat([[batch_size], [max_seq_length]], 0), dtype=dtype)
    else:
        mask = tf.sequence_mask(sequence_length, dtype=dtype)
        mask = tf.concat([mask[:, 1:], tf.zeros_like(mask[:, -1:])], axis=1)
        # Make each row = [discount, ..., discount, 1, ..., 1]
        dmat = mask * discount + (1 - mask)
        dmat = tf.cumprod(dmat, axis=1, reverse=True)

    disc_reward = dmat * tf.expand_dims(reward, -1)
    disc_reward = mask_sequences(
        disc_reward, sequence_length, dtype=dtype, tensor_rank=2)

    return disc_reward

def _discount_reward_py_2d(reward, sequence_length=None,
                           discount=1., dtype=None):
    if sequence_length is not None:
        reward = mask_sequences(reward, sequence_length, dtype=dtype)

    dtype = dtype or reward.dtype

    if discount == 1.:
        disc_reward = np.cumsum(
            reward[:, ::-1], axis=1, dtype=dtype)[:, ::-1]
    else:
        disc_reward = np.copy(reward)
        for i in range(reward.shape[1]-2, -1, -1):
            disc_reward[:, i] += disc_reward[:, i+1] * discount

    return disc_reward

def _discount_reward_tensor_2d(reward, sequence_length=None,
                               discount=1., dtype=None):
    if sequence_length is not None:
        reward = mask_sequences(
            reward, sequence_length, dtype=dtype, tensor_rank=2)

    if discount == 1.:
        disc_reward = tf.cumsum(reward, axis=1, reverse=True)
    else:
        # [max_time, batch_size]
        rev_reward_T = tf.transpose(tf.reverse(reward, [1]), [1, 0])
        rev_reward_T_cum = tf.scan(
            fn=lambda acc, cur: cur + discount * acc,
            elems=rev_reward_T,
            initializer=tf.zeros_like(reward[:, 1]),
            back_prop=False)
        disc_reward = tf.reverse(
            tf.transpose(rev_reward_T_cum, [1, 0]), [1])

    return disc_reward
