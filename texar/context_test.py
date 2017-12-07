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

class ContextTest(tf.test.TestCase):
    """Tests context.
    """

    def test_is_train(self):
        """Tests the mode context manager.
        """
        is_train = context.is_train()
        self.assertIsInstance(is_train, tf.Tensor)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            is_train_ = sess.run(is_train,
                                 feed_dict={context.is_train(): True})
            self.assertTrue(is_train_)

            is_train_ = sess.run(is_train,
                                 feed_dict={context.is_train(): False})
            self.assertFalse(is_train_)

        is_train_values = tf.get_collection_ref(context._IS_TRAIN_KEY)  # pylint: disable=protected-access
        self.assertEqual(len(is_train_values), 1)


if __name__ == "__main__":
    tf.test.main()
