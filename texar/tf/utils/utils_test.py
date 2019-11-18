"""
Unit tests for utility functions.
"""

import tensorflow as tf

from texar.tf.utils import utils


class UtilsTest(tf.test.TestCase):
    """Tests utility functions.
    """
    def test_uniquify_str(self):
        """Tests :func:`texar.tf.utils.uniquify_str`.
        """

        str_set = ['str']
        unique_str = utils.uniquify_str('str', str_set)
        self.assertEqual(unique_str, 'str_1')

        str_set.append('str_1')
        str_set.append('str_2')
        unique_str = utils.uniquify_str('str', str_set)
        self.assertEqual(unique_str, 'str_3')


if __name__ == "__main__":
    tf.test.main()
