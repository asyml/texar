# -*- coding: utf-8 -*-
#
"""
Unit tests for data utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile
import numpy as np

import tensorflow as tf

from texar.data.data import dataset_utils as dsutils


class TransformationTest(tf.test.TestCase):
    """Tests various transformation utilities.
    """

    def test_make_chained_transformation(self):
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

class CountFileLinesTest(tf.test.TestCase):
    """Tests :func:`texar.data.data.dsutils.count_file_lines`.
    """

    def test_load_glove(self):
        """Tests the load_glove function.
        """
        file_1 = tempfile.NamedTemporaryFile(mode="w+")
        num_lines = dsutils.count_file_lines(file_1.name)
        self.assertEqual(num_lines, 0)

        file_2 = tempfile.NamedTemporaryFile(mode="w+")
        file_2.write('\n'.join(['x']*5))
        file_2.flush()
        num_lines = dsutils.count_file_lines(
            [file_1.name, file_2.name, file_2.name])
        self.assertEqual(num_lines, 0+5+5)


if __name__ == "__main__":
    tf.test.main()

