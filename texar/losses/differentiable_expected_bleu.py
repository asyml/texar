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
    "differentiable_expected_bleu",
]

def differentiable_expected_bleu(labels,
                                 probs,
                                 sequence_length,
                                 time_major=False,
                                 min_fn=lambda x: tf.minimum(1., x),
                                 max_order=4,
                                 weights=[.1, .3, .3, .3],
                                 smooth_add=1e-9,
                                 name=None):
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
    with tf.name_scope(name, "sequence_sparse_softmax_cross_entropy"):
        X = probs
        Y = labels

        if time_major:
            X = tf.transpose(X, [1, 0, 2])
            Y = tf.transpose(Y, [1, 0])
        
        sizeX = tf.shape(X)[1]
        sizeY = tf.shape(Y)[1]

        XY = tf.batch_gather(X, tf.tile(tf.expand_dims(tf.to_int32(Y), 1), [1, sizeX, 1]))
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
            matchXY = XY[:, : sizeX - order, : sizeY - order] * matchXY[:, 1:, 1:]
            matchYY = YY[:, : sizeY - order, : sizeY - order] * matchYY[:, 1:, 1:]
            cntYX = tf.reduce_sum(matchXY, 1, keepdims=True)
            cntYY = tf.reduce_sum(matchYY, 1, keepdims=True)
            o_order = tf.reduce_sum(tf.reduce_sum(
                min_fn(cntYY / (cntYX - matchXY + 1))
                * matchXY / tf.maximum(1., cntYY),
                2), 1)
            # in order to avoid dividing 0
            tot_order = tf.maximum(1, sequence_length - order)
            tot.append(tot_order)
            o.append(o_order)

        tot = tf.stack(tot, 1)
        o = tf.stack(o, 1)
        prec = tf.reduce_sum(o, 0) / tf.to_float(tf.reduce_sum(tot, 0))
        neglog_prec = -tf.log(prec + smooth_add)
        loss = tf.reduce_sum(weights * neglog_prec, 0)
        
        return loss
