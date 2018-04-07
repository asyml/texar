# -*- coding: utf-8 -*-
#
"""
Unit tests for data utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf

from texar.data.data import dataset_utils as dsutils


# pylint: disable=invalid-name

class TransformationTest(tf.test.TestCase):
    """Tests various transformation utilities.
    """

    def test_make_chained_transformation(self):
        """Tests :func:`texar.data.make_chained_transformation`
        """
        original_data = np.arange(0, 10)
        dataset = tf.data.Dataset.from_tensor_slices(original_data)

        def _tran_a(data):
            return data + 100
        def _tran_b(data):
            return data + 1000
        def _tran_c(data):
            return data + 10000

        chained_tran = dsutils.make_chained_transformation(
            [_tran_a, _tran_b, _tran_c])
        dataset = dataset.map(chained_tran)

        iterator = dataset.make_one_shot_iterator()
        elem = iterator.get_next()
        with self.test_session() as sess:
            data_ = []
            while True:
                try:
                    data_.append(sess.run(elem))
                except tf.errors.OutOfRangeError:
                    break
            self.assertEqual(len(data_), len(original_data))
            data_ = [elem_ - 11100 for elem_ in data_]
            self.assertEqual(data_, original_data.tolist())

if __name__ == "__main__":
    tf.test.main()

