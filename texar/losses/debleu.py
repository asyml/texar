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
Differentiable Expected BLEU loss
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# pylint: disable=invalid-name, not-context-manager, protected-access,
# pylint: disable=too-many-arguments

__all__ = [
    "debleu",
]

def batch_gather(params, indices, name=None):
    """This function is copied and modified from tensorflow 11.0. See
    https://www.tensorflow.org/api_docs/python/tf/batch_gather for details.
    Gather slices from `params` according to `indices` with leading batch dims.
    This operation assumes that the leading dimensions of `indices` are dense,
    and the gathers on the axis corresponding to the last dimension of `indices`.
    More concretely it computes:
    result[i1, ..., in] = params[i1, ..., in-1, indices[i1, ..., in]]
    Therefore `params` should be a Tensor of shape [A1, ..., AN, B1, ..., BM],
    `indices` should be a Tensor of shape [A1, ..., AN-1, C] and `result` will be
    a Tensor of size `[A1, ..., AN-1, C, B1, ..., BM]`.
    In the case in which indices is a 1D tensor, this operation is equivalent to
    `tf.gather`.
    See also `tf.gather` and `tf.gather_nd`.
    Args:
      params: A Tensor. The tensor from which to gather values.
      indices: A Tensor. Must be one of the following types: int32, int64. Index
          tensor. Must be in range `[0, params.shape[axis]`, where `axis` is the
          last dimension of `indices` itself.
      name: A name for the operation (optional).
    Returns:
      A Tensor. Has the same type as `params`.
    Raises:
      ValueError: if `indices` has an unknown shape.
    """

    with tf.name_scope(name):
        indices = tf.convert_to_tensor(indices, name="indices")
        params = tf.convert_to_tensor(params, name="params")
        indices_shape = tf.shape(indices)
        params_shape = tf.shape(params)

        ndims = indices.shape.ndims
        if ndims is None:
            raise ValueError("batch_gather does not allow indices with unknown "
                             "shape.")
        batch_indices = indices
        indices_dtype = indices.dtype.base_dtype
        accum_dim_value = tf.ones((), dtype=indices_dtype)
        # Use correct type for offset index computation
        casted_params_shape = tf.cast(params_shape, indices_dtype)
        for dim in range(ndims-1, 0, -1):
            dim_value = casted_params_shape[dim-1]
            accum_dim_value *= casted_params_shape[dim]
            start = tf.zeros((), dtype=indices_dtype)
            step = tf.ones((), dtype=indices_dtype)
            dim_indices = tf.range(start, dim_value, step)
            dim_indices *= accum_dim_value
            dim_shape = tf.stack(
                [1] * (dim - 1) + [dim_value] + [1] * (ndims - dim), axis=0)
            batch_indices += tf.reshape(dim_indices, dim_shape)

        flat_indices = tf.reshape(batch_indices, [-1])
        outer_shape = params_shape[ndims:]
        flat_inner_shape = tf.reduce_prod(params_shape[:ndims])

        flat_params = tf.reshape(
            params, tf.concat([[flat_inner_shape], outer_shape], axis=0))
        flat_result = tf.gather(flat_params, flat_indices)
        result = tf.reshape(
            flat_result, tf.concat([indices_shape, outer_shape], axis=0))
        final_shape = indices.get_shape()[:ndims-1].merge_with(
            params.get_shape()[:ndims -1])
        final_shape = final_shape.concatenate(indices.get_shape()[ndims-1])
        final_shape = final_shape.concatenate(params.get_shape()[ndims:])
        result.set_shape(final_shape)
        return result

def debleu(labels, probs, sequence_length, time_major=False,
           min_fn=lambda x: tf.minimum(1., x), max_order=4,
           weights=[.1, .3, .3, .3], epsilon=1e-9, name=None):
    """Computes sparse softmax cross entropy for each time step of sequence
    predictions.

    Args:
        labels: Target class indexes. I.e., classes are mutually exclusive
            (each entry is in exactly one class).

            - If :attr:`time_major` is `False` (default), this must be\
            a Tensor of shape `[batch_size, max_time]`.

            - If `time_major` is `True`, this must be a Tensor of shape\
            `[max_time, batch_size].`
        logits: Unscaled log probabilities. This must have the shape of
            `[max_time, batch_size, num_classes]` or
            `[batch_size, max_time, num_classes]` according to
            the value of `time_major`.
        sequence_length: A Tensor of shape `[batch_size]`. Time steps beyond
            the respective sequence lengths will have zero losses.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`labels` and :attr:`logits` must have shape
            `[max_time, batch_size, ...]`. If `False`
            (default), they must have shape `[batch_size, max_time, ...]`.
        name (str, optional): A name for the operation.

    Returns:
        A Tensor containing the loss of rank 0.

    Example:

        .. code-block:: python

            embedder = WordEmbedder(vocab_size=data.vocab.size)
            decoder = BasicRNNDecoder(vocab_size=data.vocab.size)
            outputs, _, _ = decoder(
                decoding_strategy='train_greedy',
                inputs=embedder(data_batch['text_ids']),
                sequence_length=data_batch['length']-1)

            loss = sequence_sparse_softmax_cross_entropy(
                labels=data_batch['text_ids'][:, 1:],
                logits=outputs.logits,
                sequence_length=data_batch['length']-1)

    """ # TODO: rewrite example
    with tf.name_scope(name, "debleu"):
        X = probs
        Y = labels

        if time_major:
            X = tf.transpose(X, [1, 0, 2])
            Y = tf.transpose(Y, [1, 0])
        
        sizeX = tf.shape(X)[1]
        sizeY = tf.shape(Y)[1]

        XY = batch_gather(X, tf.tile(tf.expand_dims(Y, 1), [1, sizeX, 1]))
        YY = tf.to_float(tf.equal(tf.expand_dims(Y, 2), tf.expand_dims(Y, 1)))

        maskX = tf.sequence_mask(
            sequence_length + 1, maxlen=sizeX + 1, dtype=tf.float32)
        maskY = tf.sequence_mask(
            sequence_length + 1, maxlen=sizeY + 1, dtype=tf.float32)
        matchXY = tf.expand_dims(maskX, 2) * tf.expand_dims(maskY, 1)
        matchYY = tf.minimum(tf.expand_dims(maskY, 2),
                             tf.expand_dims(maskY, 1))

        tot = []
        o = []

        for order in range(max_order):
            matchXY = XY[:, : sizeX - order, : sizeY - order] * \
                      matchXY[:, 1:, 1:]
            matchYY = YY[:, : sizeY - order, : sizeY - order] * \
                      matchYY[:, 1:, 1:]
            cntYX = tf.reduce_sum(matchXY, 1, keepdims=True)
            cntYY = tf.reduce_sum(matchYY, 1, keepdims=True)
            o_order = tf.reduce_sum(tf.reduce_sum(
                min_fn(cntYY / (cntYX - matchXY + 1))
                * matchXY / tf.maximum(1., cntYY),
                2), 1)
            # in order to avoid being divided by 0
            tot_order = tf.maximum(1, sequence_length - order)
            tot.append(tot_order)
            o.append(o_order)

        tot = tf.stack(tot, 1)
        o = tf.stack(o, 1)
        prec = tf.reduce_sum(o, 0) / tf.to_float(tf.reduce_sum(tot, 0))
        neglog_prec = -tf.log(prec + epsilon)
        loss = tf.reduce_sum(weights * neglog_prec, 0)
        
        return loss
