"""
Unit tests for data utils.
"""

import numpy as np

import tensorflow as tf

import texar.tf.data.data.dataset_utils as dsutils


class TransformationTest(tf.test.TestCase):
    """Tests various transformation utilities.
    """

    def test_make_chained_transformation(self):
        """Tests :func:`texar.tf.data.make_chained_transformation`
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

        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

        data_ = []
        while True:
            try:
                elem = iterator.get_next()
                data_.append(elem)
            except tf.errors.OutOfRangeError:
                break

        self.assertEqual(len(data_), len(original_data))
        data_ = [elem_ - 11100 for elem_ in data_]
        self.assertEqual(data_, original_data.tolist())


if __name__ == "__main__":
    tf.test.main()
