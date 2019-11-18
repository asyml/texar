"""
Unit tests for mode-related utility functions.
"""

import tensorflow as tf

from texar.tf.utils import mode


class UtilsTest(tf.test.TestCase):
    r"""Tests utility functions.
    """

    def test_mode(self):
        r""" Tests mode related utilities.
        """
        training = mode.is_train_mode(None)
        self.assertTrue(training)

        training = mode.is_train_mode('train')
        self.assertTrue(training)

        training = mode.is_train_mode('eval')
        self.assertFalse(training)

        infering = mode.is_eval_mode(None)
        self.assertFalse(infering)

        infering = mode.is_eval_mode('eval')
        self.assertTrue(infering)

        infering = mode.is_predict_mode(None)
        self.assertFalse(infering)

        infering = mode.is_predict_mode('infer')
        self.assertTrue(infering)


if __name__ == "__main__":
    tf.test.main()
