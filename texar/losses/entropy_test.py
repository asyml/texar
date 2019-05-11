# -*- coding: utf-8 -*-
#
"""
Unit tests for entropy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# pylint: disable=invalid-name

import tensorflow as tf
import texar as tx


class EntropyTest(tf.test.TestCase):
    """Tests entropy.
    """

    def setUp(self):
        tf.test.TestCase.setUp(self)
        self._batch_size = 64
        self._max_time = 128
        self._d = 16
        self._distribution_dim = 32
        self._logits = tf.random_uniform([self._batch_size, self._d,
                                          self._distribution_dim])
        self._sequence_logits = tf.random_uniform([self._batch_size,
                                                   self._max_time,
                                                   self._d,
                                                   self._distribution_dim])
        self._sequence_length = tf.random_uniform(
            [self._batch_size], maxval=self._max_time, dtype=tf.int32)

    def _test_entropy(self, entropy_fn, logits, sequence_length=None):
        with self.test_session() as sess:
            if sequence_length is None:
                entropy = entropy_fn(logits)
                rank = sess.run(tf.rank(entropy))
                self.assertEqual(rank, 0)

                entropy = entropy_fn(logits, average_across_batch=False)
                rank = sess.run(tf.rank(entropy))
                self.assertEqual(rank, 1)
                self.assertEqual(entropy.shape,
                                 tf.TensorShape([self._batch_size]))
            else:
                entropy = entropy_fn(logits, sequence_length=sequence_length)
                rank = sess.run(tf.rank(entropy))
                self.assertEqual(rank, 0)

                entropy = entropy_fn(logits, sequence_length=sequence_length,
                                     sum_over_timesteps=False)
                rank = sess.run(tf.rank(entropy))
                self.assertEqual(rank, 1)
                self.assertEqual(entropy.shape,
                                 tf.TensorShape([self._max_time]))

                entropy = entropy_fn(logits, sequence_length=sequence_length,
                                     sum_over_timesteps=False,
                                     average_across_timesteps=True,
                                     average_across_batch=False)
                rank = sess.run(tf.rank(entropy))
                self.assertEqual(rank, 1)
                self.assertEqual(entropy.shape,
                                 tf.TensorShape([self._batch_size]))

                entropy = entropy_fn(logits, sequence_length=sequence_length,
                                     sum_over_timesteps=False,
                                     average_across_batch=False)
                rank = sess.run(tf.rank(entropy))
                self.assertEqual(rank, 2)
                self.assertEqual(entropy.shape,
                                 tf.TensorShape([self._batch_size,
                                                 self._max_time]))

                sequence_length_time = tf.random_uniform(
                    [self._max_time], maxval=self._batch_size, dtype=tf.int32)
                entropy = entropy_fn(logits,
                                     sequence_length=sequence_length_time,
                                     sum_over_timesteps=False,
                                     average_across_batch=False,
                                     time_major=True)
                self.assertEqual(entropy.shape, tf.TensorShape(
                    [self._batch_size, self._max_time]))

    def test_entropy_with_logits(self):
        """Tests `entropy_with_logits`
        """
        self._test_entropy(
            tx.losses.entropy_with_logits, self._logits)

    def test_sequence_entropy_with_logits(self):
        """Tests `sequence_entropy_with_logits`
        """
        self._test_entropy(
            tx.losses.sequence_entropy_with_logits, self._sequence_logits,
            sequence_length=self._sequence_length)


if __name__ == "__main__":
    tf.test.main()
