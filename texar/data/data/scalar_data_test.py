# -*- coding: utf-8 -*-
#
"""
Unit tests for data related operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import tempfile
import numpy as np

import tensorflow as tf

import texar as tx

class ScalarDataTest(tf.test.TestCase):
    """Tests scalar data class.
    """

    def setUp(self):
        tf.test.TestCase.setUp(self)

        # Create test data
        # pylint: disable=no-member
        int_data = np.linspace(0, 100, num=101, dtype=np.int32).tolist()
        int_data = [str(i) for i in int_data]
        int_file = tempfile.NamedTemporaryFile()
        int_file.write('\n'.join(int_data).encode("utf-8"))
        int_file.flush()
        self._int_file = int_file

        self._int_hparams = {
            "num_epochs": 1,
            "batch_size": 1,
            "shuffle": False,
            "dataset": {
                "files": self._int_file.name,
                "data_type": "int",
                "data_name": "label"
            }
        }

        self._float_hparams = {
            "num_epochs": 1,
            "batch_size": 1,
            "shuffle": False,
            "dataset": {
                "files": self._int_file.name,
                "data_type": "float",
                "data_name": "feat"
            }
        }


    def _run_and_test(self, hparams):
        # Construct database
        scalar_data = tx.data.ScalarData(hparams)

        self.assertEqual(scalar_data.list_items()[0],
                         hparams["dataset"]["data_name"])

        iterator = scalar_data.dataset.make_initializable_iterator()
        data_batch = iterator.get_next()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            sess.run(iterator.initializer)

            i = 0
            while True:
                try:
                    # Run the logics
                    data_batch_ = sess.run(data_batch)
                    self.assertEqual(set(data_batch_.keys()),
                                     set(scalar_data.list_items()))
                    value = data_batch_[scalar_data.data_name][0]
                    self.assertEqual(i, value)
                    i += 1
                    # pylint: disable=no-member
                    if hparams["dataset"]["data_type"] == "int":
                        self.assertTrue(isinstance(value, np.int32))
                    else:
                        self.assertTrue(isinstance(value, np.float32))
                except tf.errors.OutOfRangeError:
                    print('Done -- epoch limit reached')
                    break

    def test_default_setting(self):
        """Tests the logics of ScalarData.
        """
        self._run_and_test(self._int_hparams)
        self._run_and_test(self._float_hparams)

    def test_shuffle(self):
        """Tests results of toggling shuffle.
        """
        hparams = copy.copy(self._int_hparams)
        hparams["batch_size"] = 10
        scalar_data = tx.data.ScalarData(hparams)
        iterator = scalar_data.dataset.make_initializable_iterator()
        data_batch = iterator.get_next()

        hparams_sfl = copy.copy(hparams)
        hparams_sfl["shuffle"] = True
        scalar_data_sfl = tx.data.ScalarData(hparams_sfl)
        iterator_sfl = scalar_data_sfl.dataset.make_initializable_iterator()
        data_batch_sfl = iterator_sfl.get_next()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            sess.run(iterator.initializer)
            sess.run(iterator_sfl.initializer)

            vals = []
            vals_sfl = []
            while True:
                try:
                    # Run the logics
                    data_batch_, data_batch_sfl_ = sess.run([data_batch,
                                                             data_batch_sfl])
                    vals += data_batch_[scalar_data.data_name].tolist()
                    vals_sfl += data_batch_sfl_[scalar_data.data_name].tolist()
                except tf.errors.OutOfRangeError:
                    print('Done -- epoch limit reached')
                    break
            self.assertEqual(len(vals), len(vals_sfl))
            self.assertSetEqual(set(vals), set(vals_sfl))

if __name__ == "__main__":
    tf.test.main()
