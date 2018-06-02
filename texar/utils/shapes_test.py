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

if __name__ == "__main__":
    tf.test.main()

