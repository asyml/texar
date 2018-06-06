# -*- coding: utf-8 -*-
#
"""
Unit tests for various context functionalities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from texar import context

# pylint: disable=protected-access

class ContextTest(tf.test.TestCase):
    """Tests context.
    """

    def test_global_mode(self):
        """Tests the mode context manager.
        """
        global_mode = context.global_mode()
        self.assertIsInstance(global_mode, tf.Tensor)

        mode_train = context.global_mode_train()
        mode_eval = context.global_mode_eval()
        mode_predict = context.global_mode_predict()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            global_mode_ = sess.run(global_mode)
            self.assertEqual(tf.compat.as_str(global_mode_),
                             tf.estimator.ModeKeys.TRAIN)

            global_mode_, mode_train_, mode_eval_, mode_predict_ = sess.run(
                [global_mode, mode_train, mode_eval, mode_predict],
                feed_dict={context.global_mode(): tf.estimator.ModeKeys.TRAIN})
            self.assertEqual(global_mode_, tf.estimator.ModeKeys.TRAIN)
            self.assertTrue(mode_train_)
            self.assertFalse(mode_eval_)
            self.assertFalse(mode_predict_)

            global_mode_, mode_train_, mode_eval_, mode_predict_ = sess.run(
                [global_mode, mode_train, mode_eval, mode_predict],
                feed_dict={context.global_mode(): tf.estimator.ModeKeys.EVAL})
            self.assertEqual(global_mode_, tf.estimator.ModeKeys.EVAL)
            self.assertFalse(mode_train_)
            self.assertTrue(mode_eval_)
            self.assertFalse(mode_predict_)

            global_mode_, mode_train_, mode_eval_, mode_predict_ = sess.run(
                [global_mode, mode_train, mode_eval, mode_predict],
                feed_dict={context.global_mode():
                           tf.estimator.ModeKeys.PREDICT})
            self.assertEqual(global_mode_, tf.estimator.ModeKeys.PREDICT)
            self.assertFalse(mode_train_)
            self.assertFalse(mode_eval_)
            self.assertTrue(mode_predict_)

        global_mode_values = tf.get_collection_ref(context._GLOBAL_MODE_KEY)
        self.assertEqual(len(global_mode_values), 1)

if __name__ == "__main__":
    tf.test.main()
