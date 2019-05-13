# -*- coding: utf-8 -*-
#
"""
Unit tests for pg losses.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# pylint: disable=invalid-name

import tensorflow as tf
import texar as tx


class PGLossesTest(tf.test.TestCase):
    """Tests pg losses.
    """

    def setUp(self):
        tf.test.TestCase.setUp(self)
        self._batch_size = 64
        self._max_time = 16
        self._d1 = 32
        self._d2 = 32
        self._d3 = 32
        self._num_classes = 10

        self._actions_batch = tf.ones([self._batch_size, self._max_time,
                                      self._d1, self._d2, self._d3],
                                      dtype=tf.int32)
        self._logits_batch = tf.random_uniform([self._batch_size,
                                                self._max_time,
                                                self._d1, self._d2, self._d3,
                                                self._num_classes])
        self._advantages_batch = tf.random_uniform([self._batch_size,
                                                    self._max_time,
                                                    self._d1, self._d2,
                                                    self._d3])

        self._sequence_length = tf.random_uniform(
            [self._batch_size], maxval=self._max_time, dtype=tf.int32)

    def _test_sequence_loss(self, loss_fn, actions, logits, advantages, rank,
                            batched, sequence_length):
        with self.test_session() as sess:
            loss = loss_fn(actions, logits, advantages, rank, batched,
                           sequence_length)
            rank = sess.run(tf.rank(loss))
            self.assertEqual(rank, 0)

    def test_pg_losses_with_logits(self):
        """Tests `texar.losses.pg_losses_with_logits`.
        """
        self._test_sequence_loss(tx.losses.pg_loss_with_logits,
                                 self._actions_batch,
                                 self._logits_batch,
                                 self._advantages_batch,
                                 None,
                                 True,
                                 self._sequence_length)


if __name__ == "__main__":
    tf.test.main()






















