
"""
Unit tests for mode-related utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.utils import mode
from texar import context


class UtilsTest(tf.test.TestCase):
    """Tests utility functions.
    """

    def test_mode(self):
        """ Tests mode related utilities.
        """
        training = mode.is_train_mode(None)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            training_ = sess.run(training)
            self.assertTrue(training_)

            training_ = sess.run(
                training,
                feed_dict={context.global_mode(): tf.estimator.ModeKeys.TRAIN})
            self.assertTrue(training_)

            training_ = sess.run(
                training,
                feed_dict={context.global_mode(): tf.estimator.ModeKeys.EVAL})
            self.assertFalse(training_)

        training = mode.is_train_mode(tf.estimator.ModeKeys.TRAIN)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            training_ = sess.run(training)
            self.assertTrue(training_)


if __name__ == "__main__":
    tf.test.main()

