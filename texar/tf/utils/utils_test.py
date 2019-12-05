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

    def test_truncate_seq_pair(self):

        tokens_a = [1, 2, 3]
        tokens_b = [4, 5, 6]
        utils.truncate_seq_pair(tokens_a, tokens_b, 4)
        self.assertListEqual(tokens_a, [1, 2])
        self.assertListEqual(tokens_b, [4, 5])

        tokens_a = [1]
        tokens_b = [2, 3, 4, 5]
        utils.truncate_seq_pair(tokens_a, tokens_b, 3)
        self.assertListEqual(tokens_a, [1])
        self.assertListEqual(tokens_b, [2, 3])


if __name__ == "__main__":
    tf.test.main()
