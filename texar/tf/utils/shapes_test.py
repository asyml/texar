"""
Unit tests for shape-related utility functions.
"""

# pylint: disable=no-member

import numpy as np

import tensorflow as tf

from texar.tf.utils import shapes


class ShapesTest(tf.test.TestCase):
    """Tests shape-related utility functions.
    """

    def test_mask_sequences(self):
        """Tests :func:`texar.tf.utils.shapes.mask_sequences`.
        """
        seq = np.ones([3, 4, 3], dtype=np.int32)
        seq_length = np.array([3, 2, 1], dtype=np.int32)

        masked_seq = shapes.mask_sequences(seq, seq_length)
        self.assertEqual(masked_seq.shape, seq.shape)
        seq_sum = np.sum(masked_seq, axis=(1, 2))
        np.testing.assert_array_equal(seq_sum, seq_length * 3)

    def test_reduce_with_weights(self):
        """Tests :func:`texar.tf.utils.shapes.reduce_with_weights`
        """
        x = np.asarray([[10, 10, 2, 2],
                        [20, 2, 2, 2]])
        x = tf.constant(x)
        w = np.asarray([[1, 1, 0, 0],
                        [1, 0, 0, 0]])

        z = shapes.reduce_with_weights(x, weights=w)

        with self.test_session() as sess:
            z_ = sess.run(z)
            np.testing.assert_array_equal(z_, 20)

    def test_pad_and_concat(self):
        """Test :func:`texar.tf.utils.shapes.pad_and_concat`.
        """
        a = tf.ones([3, 10, 2])
        b = tf.ones([4, 20, 3])
        c = tf.ones([5, 1, 4])

        t = shapes.pad_and_concat([a, b, c], 0)
        self.assertEqual(t.shape, [3 + 4 + 5, 20, 4])
        t = shapes.pad_and_concat([a, b, c], 1)
        self.assertEqual(t.shape, [5, 10 + 20 + 1, 4])
        t = shapes.pad_and_concat([a, b, c], 2)
        self.assertEqual(t.shape, [5, 20, 2 + 3 + 4])

        d = tf.placeholder(dtype=tf.float32, shape=[6, None, 1])
        t = shapes.pad_and_concat([a, b, c, d], 0)
        with self.test_session() as sess:
            t_ = sess.run(t, feed_dict={d: np.ones([6, 2, 1])})
            self.assertEqual(list(t_.shape), [3 + 4 + 5 + 6, 20, 4])

    def test_varlength_concat(self):
        """
        Tests :func:`texar.tf.utils.shapes.varlength_concat`.
        """
        # 2D
        x = np.asarray(
            [[1, 1, 0, 0],
             [1, 0, 0, 0],
             [1, 1, 1, 1]], dtype=np.int32)
        x_length = np.asarray([2, 1, 4], dtype=np.int32)
        y = np.asarray(
            [[2, 2, 2, 0],
             [2, 2, 2, 2],
             [2, 2, 0, 0]], dtype=np.int32)

        z_true = np.asarray(
            [[1, 1, 2, 2, 2, 0, 0, 0],
             [1, 2, 2, 2, 2, 0, 0, 0],
             [1, 1, 1, 1, 2, 2, 0, 0]], dtype=np.int32)

        # py
        z = shapes.varlength_concat_py(x, y, x_length)
        np.testing.assert_array_equal(z, z_true)

        # tf
        z = shapes.varlength_concat(x, y, x_length)
        with self.test_session() as sess:
            z_ = sess.run(z)
            np.testing.assert_array_equal(z_, z_true)

        # 3D
        x = np.asarray(
            [[[1], [1], [0], [0]],
             [[1], [0], [0], [0]],
             [[1], [1], [1], [1]]], dtype=np.int32)
        x_length = [2, 1, 4]
        y = np.asarray(
            [[[2], [2], [2], [0]],
             [[2], [2], [2], [2]],
             [[2], [2], [0], [0]]], dtype=np.int32)
        z_true = np.asarray(
            [[[1], [1], [2], [2], [2], [0], [0], [0]],
             [[1], [2], [2], [2], [2], [0], [0], [0]],
             [[1], [1], [1], [1], [2], [2], [0], [0]]], dtype=np.int32)

        # py
        z = shapes.varlength_concat_py(x, y, x_length)
        np.testing.assert_array_equal(z, z_true)

        # tf
        z = shapes.varlength_concat(x, y, x_length)
        with self.test_session() as sess:
            z_ = sess.run(z)
            np.testing.assert_array_equal(z_, z_true)

    def test_varlength_roll(self):
        """
        Tests :func:`texar.tf.utils.shapes.varlength_roll`.
        """
        # 2D
        x = np.asarray(
            [[1, 1, 0, 0],
             [1, 0, 0, 0],
             [1, 1, 1, 1]], dtype=np.int32)
        x_length = [-2, -1, -4]
        z = shapes.varlength_roll(x, x_length)

        with self.test_session() as sess:
            z_ = sess.run(z)

            z_true = np.asarray(
                [[0, 0, 1, 1],
                 [0, 0, 0, 1],
                 [1, 1, 1, 1]], dtype=np.int32)

            np.testing.assert_array_equal(z_, z_true)

        # 3D
        x = np.asarray(
            [[[1], [1], [0], [0]],
             [[1], [0], [0], [0]],
             [[1], [1], [1], [1]]], dtype=np.int32)
        x_length = [-2, -1, -4]
        z = shapes.varlength_roll(x, x_length)

        with self.test_session() as sess:
            z_ = sess.run(z)

            z_true = np.asarray(
                [[[0], [0], [1], [1]],
                 [[0], [0], [0], [1]],
                 [[1], [1], [1], [1]]], dtype=np.int32)

            np.testing.assert_array_equal(z_, z_true)


if __name__ == "__main__":
    tf.test.main()
