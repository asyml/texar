# -*- coding: utf-8 -*-
#
"""
Unit tests for mle losses.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# pylint: disable=invalid-name

import numpy as np

import tensorflow as tf

import texar as tx

class MLELossesTest(tf.test.TestCase):
    """Tests mle losses.
    """

    def setUp(self):
        tf.test.TestCase.setUp(self)
        self._batch_size = 64
        self._max_time = 16
        self._num_classes = 100
        self._labels = tf.ones([self._batch_size, self._max_time],
                               dtype=tf.int32)
        one_hot_labels = tf.one_hot(
            self._labels, self._num_classes, dtype=tf.float32)
        self._one_hot_labels = tf.reshape(
            one_hot_labels, [self._batch_size, self._max_time, -1])
        self._logits = tf.random_uniform(
            [self._batch_size, self._max_time, self._num_classes])
        self._sequence_length = tf.random_uniform(
            [self._batch_size], maxval=self._max_time, dtype=tf.int32)

    def _test_sequence_loss(self, loss_fn, labels, logits, sequence_length):
        with self.test_session() as sess:
            loss = loss_fn(labels, logits, sequence_length)
            rank = sess.run(tf.rank(loss))
            self.assertEqual(rank, 0)

            loss = loss_fn(
                labels, logits, sequence_length, sum_over_timesteps=False)
            rank = sess.run(tf.rank(loss))
            self.assertEqual(rank, 1)
            self.assertEqual(loss.shape, tf.TensorShape([self._max_time]))

            loss = loss_fn(
                labels, logits, sequence_length, sum_over_timesteps=False,
                average_across_timesteps=True, average_across_batch=False)
            rank = sess.run(tf.rank(loss))
            self.assertEqual(rank, 1)
            self.assertEqual(loss.shape, tf.TensorShape([self._batch_size]))

            loss = loss_fn(
                labels, logits, sequence_length, sum_over_timesteps=False,
                average_across_batch=False)
            rank = sess.run(tf.rank(loss))
            self.assertEqual(rank, 2)
            self.assertEqual(loss.shape,
                             tf.TensorShape([self._batch_size, self._max_time]))

            sequence_length_time = tf.random_uniform(
                [self._max_time], maxval=self._batch_size, dtype=tf.int32)
            loss = loss_fn(
                labels, logits, sequence_length_time, sum_over_timesteps=False,
                average_across_batch=False, time_major=True)
            self.assertEqual(loss.shape,
                             tf.TensorShape([self._batch_size, self._max_time]))

    def test_sequence_softmax_cross_entropy(self):
        """Tests `sequence_softmax_cross_entropy`
        """
        self._test_sequence_loss(
            tx.losses.sequence_softmax_cross_entropy,
            self._one_hot_labels, self._logits, self._sequence_length)

    def test_sequence_sparse_softmax_cross_entropy(self):
        """Tests `sequence_sparse_softmax_cross_entropy`
        """
        self._test_sequence_loss(
            tx.losses.sequence_sparse_softmax_cross_entropy,
            self._labels, self._logits, self._sequence_length)

    def test_sequence_sigmoid_cross_entropy(self):
        """Tests `texar.losses.test_sequence_sigmoid_cross_entropy`.
        """
        self._test_sequence_loss(
            tx.losses.sequence_sigmoid_cross_entropy,
            self._one_hot_labels, self._logits, self._sequence_length)

        self._test_sequence_loss(
            tx.losses.sequence_sigmoid_cross_entropy,
            self._one_hot_labels[:, :, 0],
            self._logits[:, :, 0],
            self._sequence_length)

        labels = tf.placeholder(dtype=tf.int32, shape=None)
        loss = tx.losses.sequence_sigmoid_cross_entropy(
            logits=self._logits[:, :, 0],
            labels=tf.to_float(labels),
            sequence_length=self._sequence_length)
        with self.test_session() as sess:
            rank = sess.run(
                tf.rank(loss),
                feed_dict={labels: np.ones([self._batch_size, self._max_time])})
            self.assertEqual(rank, 0)

if __name__ == "__main__":
    tf.test.main()
