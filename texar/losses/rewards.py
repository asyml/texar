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
    "discount_reward",
    "_discount_reward_py_1d",
    "_discount_reward_tensor_1d",
    "_discount_reward_py_2d",
    "_discount_reward_tensor_2d"
]

def discount_reward(reward,
                    sequence_length=None,
                    discount=1.,
                    normalize=False,
                    dtype=None,
                    tensor_rank=1):
    """Computes discounted reward.

    :attr:`reward` and :attr:`sequence_length` can be either Tensors or python
    arrays. If both are python array (or `None`), the return will be a python
    array as well. Otherwise tf Tensors are returned.

    Args:
        reward: A Tensor or python array. Can be 1D with shape `[batch_size]`,
            or 2D with shape `[batch_size, max_time]`.
        sequence_length (optional): A Tensor or python array of shape
            `[batch_size]`. Time steps beyond the respective sequence lengths
            will be masked. Required if :attr:`reward` is 1D.
        discount (float): A scalar. The discount factor.
        normalize (bool): Whether to normalize the discounted reward, by
            `(discounted_reward - mean) / std`. Here `mean` and `std` are
            over all time steps and all samples in the batch.
        dtype (dtype): Type of :attr:`reward`. If `None`, infer from
            `reward` automatically.
        tensor_rank (int): The number of dimensions of :attr:`reward`.
            Default is 1, i.e., :attr:`reward` is a 1D Tensor consisting
            of a batch dimension. Ignored if :attr:`reward`
            and :attr:`sequence_length` are python arrays (or `None`).

    Returns:
        A 2D Tensor or python array of the discounted reward.

        If :attr:`reward` and :attr:`sequence_length` are python
        arrays (or `None`), the returned value is a python array as well.


    Example:

        .. code-block:: python

            r = [2., 1.]
            seq_length = [3, 2]
            discounted_r = discount_reward(r, seq_length, discount=0.1)
            # discounted_r == [[2. * 0.1^2, 2. * 0.1, 2.],
            #                  [1. * 0.1,   1.,       0.]]

            r = [[3., 4., 5.], [6., 7., 0.]]
            seq_length = [3, 2]
            discounted_r = discount_reward(r, seq_length, discount=0.1)
            # discounted_r == [[3. + 4.*0.1 + 5.*0.1^2, 4. + 5.*0.1, 5.],
            #                  [6. + 7.*0.1,            7.,          0.]]
    """
    is_tensor = tf.contrib.framework.is_tensor
    if is_tensor(reward) or is_tensor(sequence_length):
        if tensor_rank == 1:
            disc_reward = _discount_reward_tensor_1d(
                reward, sequence_length, discount, dtype)
        elif tensor_rank == 2:
            disc_reward = _discount_reward_tensor_2d(
                reward, sequence_length, discount, dtype)
        else:
            raise ValueError("`tensor_rank` can only be 1 or 2.")

        if normalize:
            mu, var = tf.nn.moments(disc_reward, axes=[0, 1], keep_dims=True)
            disc_reward = (disc_reward - mu) / (tf.sqrt(var) + 1e-8)
    else:
        reward = np.array(reward)
        tensor_rank = reward.ndim
        if tensor_rank == 1:
            disc_reward = _discount_reward_py_1d(
                reward, sequence_length, discount, dtype)
        elif tensor_rank == 2:
            disc_reward = _discount_reward_py_2d(
                reward, sequence_length, discount, dtype)
        else:
            raise ValueError("`reward` can only be 1D or 2D.")

        if normalize:
            mu = np.mean(disc_reward)
            std = np.std(disc_reward)
            disc_reward = (disc_reward - mu) / (std + 1e-8)

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
