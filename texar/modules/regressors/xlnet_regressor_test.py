#
"""
Unit tests for XLNet regressor.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf

from texar.modules.regressors.xlnet_regressor import XLNetRegressor

# pylint: disable=too-many-locals, no-member


class XLNetRegressorTest(tf.test.TestCase):
    """Tests :class:`~texar.modules.XLNetRegressor` class.
    """

    def test_trainable_variables(self):
        """Tests the functionality of automatically collecting trainable
        variables.
        """
        inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])

        # case 1
        regressor = XLNetRegressor()
        regressor(inputs)
        n_xlnet_vars = 182
        n_projection_vars = 2
        n_logits_vars = 2
        self.assertEqual(len(regressor.trainable_variables),
                         n_xlnet_vars + n_logits_vars + n_projection_vars)

        # case 2
        hparams = {
            "regr_strategy": "all_time"
        }
        regressor = XLNetRegressor(hparams=hparams)
        regressor(inputs)
        self.assertEqual(len(regressor.trainable_variables),
                         n_xlnet_vars + n_logits_vars + n_projection_vars)

        # case 3
        hparams = {
            "regr_strategy": "time_wise"
        }
        regressor = XLNetRegressor(hparams=hparams)
        regressor(inputs)
        self.assertEqual(len(regressor.trainable_variables),
                         n_xlnet_vars + n_logits_vars + n_projection_vars)

    def test_encode(self):
        """Tests encoding.
        """
        max_time = 8
        batch_size = 16
        inputs = tf.random_uniform([batch_size, max_time],
                                   maxval=30521, dtype=tf.int32)

        # case 1
        regressor = XLNetRegressor()
        logits = regressor(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_ = sess.run(logits)
            self.assertEqual(logits_.shape, (batch_size,))

        # case 2
        hparams = {
            "regr_strategy": "cls_time"
        }
        regressor = XLNetRegressor(hparams=hparams)
        logits = regressor(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_ = sess.run(logits)
            self.assertEqual(logits_.shape, (batch_size,))

        # case 3
        hparams = {
            "regr_strategy": "time_wise"
        }
        regressor = XLNetRegressor(hparams=hparams)
        logits = regressor(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_ = sess.run(logits)
            self.assertEqual(logits_.shape,
                             (batch_size, max_time))

        # case 4
        hparams = {
            "regr_strategy": "all_time",
            "max_seq_len": max_time
        }
        inputs = tf.placeholder(tf.int32, shape=[batch_size, 6])
        regressor = XLNetRegressor(hparams=hparams)
        logits = regressor(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_ = sess.run(
                logits,
                feed_dict={inputs: np.random.randint(30521,
                                                     size=(batch_size, 6))})
            self.assertEqual(logits_.shape, (batch_size,))

    def test_regression(self):
        """Test the type of regression output."""
        batch_size = 8

        hparams = {
            "regr_strategy": "cls_time"
        }
        inputs = tf.placeholder(tf.int32, shape=[batch_size, 6])
        regressor = XLNetRegressor(hparams=hparams)
        logits = regressor(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_ = sess.run(
                logits,
                feed_dict={inputs: np.random.randint(30521,
                                                     size=(batch_size, 6))})
            self.assertEqual(logits_.dtype, np.float32)


if __name__ == "__main__":
    tf.test.main()
