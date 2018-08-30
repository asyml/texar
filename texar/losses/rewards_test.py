"""
Unit tests for RL rewards.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# pylint: disable=invalid-name, no-member

import numpy as np

import tensorflow as tf

from texar.losses.rewards import \
        _discount_reward_tensor_2d, _discount_reward_tensor_1d, \
        _discount_reward_py_1d, _discount_reward_py_2d, \
        discount_reward

class RewardTest(tf.test.TestCase):
    """Tests reward related functions.
    """

    def test_discount_reward(self):
        """Tests :func:`texar.losses.rewards.discount_reward`
        """
        # 1D
        reward = np.ones([2], dtype=np.float64)
        sequence_length = [3, 5]

        discounted_reward = discount_reward(
            reward, sequence_length, discount=1.)
        discounted_reward_n = discount_reward(
            reward, sequence_length, discount=.1, normalize=True)

        discounted_reward_ = discount_reward(
            tf.constant(reward, dtype=tf.float64),
            sequence_length, discount=1.)
        discounted_reward_n_ = discount_reward(
            tf.constant(reward, dtype=tf.float64),
            sequence_length, discount=.1, normalize=True)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            r, r_n = sess.run([discounted_reward_, discounted_reward_n_])

            np.testing.assert_array_almost_equal(
                discounted_reward, r, decimal=6)
            np.testing.assert_array_almost_equal(
                discounted_reward_n, r_n, decimal=6)

        # 2D
        reward = np.ones([2, 10], dtype=np.float64)
        sequence_length = [5, 10]

        discounted_reward = discount_reward(
            reward, sequence_length, discount=1.)
        discounted_reward_n = discount_reward(
            reward, sequence_length, discount=.1, normalize=True)

        discounted_reward_ = discount_reward(
            tf.constant(reward, dtype=tf.float64), sequence_length,
            discount=1., tensor_rank=2)
        discounted_reward_n_ = discount_reward(
            tf.constant(reward, dtype=tf.float64), sequence_length,
            discount=.1, tensor_rank=2, normalize=True)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            r, r_n = sess.run([discounted_reward_, discounted_reward_n_])

            np.testing.assert_array_almost_equal(
                discounted_reward, r, decimal=6)
            np.testing.assert_array_almost_equal(
                discounted_reward_n, r_n, decimal=6)

    def test_discount_reward_py_1d(self):
        """Tests :func:`texar.losses.rewards._discount_reward_py_1d`
        """
        reward = np.ones([2], dtype=np.float64)
        sequence_length = [3, 5]

        discounted_reward_1 = _discount_reward_py_1d(
            reward, sequence_length, discount=1.)

        discounted_reward_2 = _discount_reward_py_1d(
            reward, sequence_length, discount=.1)

        r = discounted_reward_1
        for i in range(5):
            if i < 3:
                self.assertEqual(r[0, i], 1)
            else:
                self.assertEqual(r[0, i], 0)
            self.assertEqual(r[1, i], 1)

        r = discounted_reward_2
        for i in range(5):
            if i < 3:
                self.assertAlmostEqual(r[0, i], 0.1**(2-i))
            else:
                self.assertAlmostEqual(r[0, i], 0)
            self.assertAlmostEqual(r[1, i], 0.1**(4-i))

    def test_discount_reward_tensor_1d(self):
        """Tests :func:`texar.losses.rewards._discount_reward_tensor_1d`
        """
        reward = tf.ones([2], dtype=tf.float64)
        sequence_length = [3, 5]

        discounted_reward_1 = _discount_reward_tensor_1d(
            reward, sequence_length, discount=1.)

        discounted_reward_2 = _discount_reward_tensor_1d(
            reward, sequence_length, discount=.1)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            r = sess.run(discounted_reward_1)
            for i in range(5):
                if i < 3:
                    self.assertEqual(r[0, i], 1)
                else:
                    self.assertEqual(r[0, i], 0)
                self.assertEqual(r[1, i], 1)

            r = sess.run(discounted_reward_2)
            for i in range(5):
                if i < 3:
                    self.assertAlmostEqual(r[0, i], 0.1**(2-i))
                else:
                    self.assertAlmostEqual(r[0, i], 0)
                self.assertAlmostEqual(r[1, i], 0.1**(4-i))

    def test_discount_reward_py_2d(self):
        """Tests :func:`texar.losses.rewards._discount_reward_py_2d`
        """
        reward = np.ones([2, 10], dtype=np.float64)
        sequence_length = [5, 10]

        discounted_reward_1 = _discount_reward_py_2d(
            reward, sequence_length, discount=1.)

        discounted_reward_2 = _discount_reward_py_2d(
            reward, sequence_length, discount=.1)

        r = discounted_reward_1
        for i in range(10):
            if i < 5:
                self.assertEqual(r[0, i], 5 - i)
            else:
                self.assertEqual(r[0, i], 0)
            self.assertEqual(r[1, i], 10 - i)

        r = discounted_reward_2
        for i in range(10):
            if i < 5:
                self.assertEqual(r[0, i], int(11111./10**i) / 10**(4-i))
            else:
                self.assertEqual(r[0, i], 0)
            self.assertEqual(r[1, i], int(1111111111./10**i) / 10**(9-i))

    def test_discount_reward_tensor_2d(self):
        """Tests :func:`texar.losses.rewards._discount_reward_tensor_2d`
        """
        reward = tf.ones([2, 10], dtype=tf.float64)
        sequence_length = [5, 10]

        discounted_reward_1 = _discount_reward_tensor_2d(
            reward, sequence_length, discount=1.)

        discounted_reward_2 = _discount_reward_tensor_2d(
            reward, sequence_length, discount=.1)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            r = sess.run(discounted_reward_1)
            for i in range(10):
                if i < 5:
                    self.assertEqual(r[0, i], 5 - i)
                else:
                    self.assertEqual(r[0, i], 0)
                self.assertEqual(r[1, i], 10 - i)

            r = sess.run(discounted_reward_2)
            for i in range(10):
                if i < 5:
                    self.assertEqual(r[0, i], int(11111./10**i) / 10**(4-i))
                else:
                    self.assertEqual(r[0, i], 0)
                self.assertEqual(r[1, i], int(1111111111./10**i) / 10**(9-i))

if __name__ == "__main__":
    tf.test.main()
