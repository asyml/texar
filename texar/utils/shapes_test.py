"""
Unit tests for shape-related utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=no-member

import numpy as np

import tensorflow as tf

from texar.utils import shapes

class ShapesTest(tf.test.TestCase):
    """Tests shape-related utility functions.
    """

    def test_mask_sequences(self):
        """Tests :func:`texar.utils.shapes.mask_sequences`.
        """
        seq = np.ones([3, 4, 3], dtype=np.int32)
        seq_length = np.array([3, 2, 1], dtype=np.int32)

        masked_seq = shapes.mask_sequences(seq, seq_length)
        self.assertEqual(masked_seq.shape, seq.shape)
        seq_sum = np.sum(masked_seq, axis=(1, 2))
        np.testing.assert_array_equal(seq_sum, seq_length * 3)

    def test_pad_and_concat(self):
        """Test :func:`texar.utils.shapes.pad_and_concat`.
        """
        a = tf.ones([3, 10, 2])
        b = tf.ones([4, 20, 3])
        c = tf.ones([5, 1, 4])

        t = shapes.pad_and_concat([a, b, c], 0)
        self.assertEqual(t.shape, [3+4+5, 20, 4])
        t = shapes.pad_and_concat([a, b, c], 1)
        self.assertEqual(t.shape, [5, 10+20+1, 4])
        t = shapes.pad_and_concat([a, b, c], 2)
        self.assertEqual(t.shape, [5, 20, 2+3+4])

        d = tf.placeholder(dtype=tf.float32, shape=[6, None, 1])
        t = shapes.pad_and_concat([a, b, c, d], 0)
        with self.test_session() as sess:
            t_ = sess.run(t, feed_dict={d: np.ones([6, 2, 1])})
            self.assertEqual(list(t_.shape), [3+4+5+6, 20, 4])

if __name__ == "__main__":
    tf.test.main()

