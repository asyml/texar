# -*- coding: utf-8 -*-
#
"""
Unit tests for data iterator related operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# pylint: disable=no-member

import tempfile
import numpy as np

import tensorflow as tf

import texar as tx


class DataIteratorTest(tf.test.TestCase):
    """Tests :class:`texar.data.DataIterator` and
    :class:`texar.data.TrainTestDataIterator`.
    """

    def setUp(self):
        tf.test.TestCase.setUp(self)

        # Create data
        train_text = list(np.linspace(1, 1000, num=1000, dtype=np.int64))
        train_text = [str(x) for x in train_text]
        train_text_file = tempfile.NamedTemporaryFile()
        train_text_file.write('\n'.join(train_text).encode("utf-8"))
        train_text_file.flush()
        self._train_text_file = train_text_file

        test_text = list(np.linspace(1001, 2000, num=1000, dtype=np.int64))
        test_text = [str(x) for x in test_text]
        test_text_file = tempfile.NamedTemporaryFile()
        test_text_file.write('\n'.join(test_text).encode("utf-8"))
        test_text_file.flush()
        self._test_text_file = test_text_file

        vocab_list = train_text + test_text
        vocab_file = tempfile.NamedTemporaryFile()
        vocab_file.write('\n'.join(vocab_list).encode("utf-8"))
        vocab_file.flush()
        self._vocab_file = vocab_file
        self._vocab_size = len(vocab_list)

        self._train_hparams = {
            "num_epochs": 2,
            "batch_size": 1,
            "shuffle": False,
            "dataset": {
                "files": self._train_text_file.name,
                "vocab_file": self._vocab_file.name,
                "bos_token": '',
                "eos_token": ''
            },
            "name": "train"
        }

        self._test_hparams = {
            "num_epochs": 1,
            "batch_size": 1,
            "shuffle": False,
            "dataset": {
                "files": self._test_text_file.name,
                "vocab_file": self._vocab_file.name,
                "bos_token": '',
                "eos_token": ''
            },
            "name": "test"
        }

    def test_iterator_single_dataset(self):
        """Tests iterating over a single dataset.
        """
        data = tx.data.MonoTextData(self._test_hparams)

        iterator = tx.data.DataIterator(data)
        data_batch = iterator.get_next()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())

            for _ in range(2):
                iterator.switch_to_dataset(sess)
                i = 1001
                while True:
                    try:
                        data_batch_ = sess.run(data_batch)
                        self.assertEqual(data_batch_['text'][0][0], str(i))
                        i += 1
                    except tf.errors.OutOfRangeError:
                        print('Done -- epoch limit reached')
                        self.assertEqual(i, 2001)
                        break


    def test_iterator_multi_datasets(self):
        """Tests iterating over multiple datasets.
        """
        train_data = tx.data.MonoTextData(self._train_hparams)
        test_data = tx.data.MonoTextData(self._test_hparams)

        iterator = tx.data.DataIterator([train_data, test_data])
        data_batch = iterator.get_next()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())

            for _ in range(2):
                # Iterates over train data
                iterator.switch_to_dataset(sess, train_data.name)
                i = 0
                while True:
                    try:
                        data_batch_ = sess.run(data_batch)
                        self.assertEqual(data_batch_['text'][0][0], str(i+1))
                        i = (i+1) % 1000
                    except tf.errors.OutOfRangeError:
                        print('Train data limit reached')
                        self.assertEqual(i, 0)
                        break

                # Iterates over test data
                iterator.switch_to_dataset(sess, test_data.name)
                i = 1001
                while True:
                    try:
                        data_batch_ = sess.run(data_batch)
                        self.assertEqual(data_batch_['text'][0][0], str(i))
                        i += 1
                    except tf.errors.OutOfRangeError:
                        print('Test data limit reached')
                        self.assertEqual(i, 2001)
                        break


    def test_train_test_data_iterator(self):
        """Tests :class:`texar.data.TrainTestDataIterator`
        """
        train_data = tx.data.MonoTextData(self._train_hparams)
        test_data = tx.data.MonoTextData(self._test_hparams)

        iterator = tx.data.TrainTestDataIterator(train=train_data,
                                                 test=test_data)
        data_batch = iterator.get_next()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())

            for _ in range(2):
                iterator.switch_to_train_data(sess)
                i = 0
                while True:
                    try:
                        data_batch_ = sess.run(data_batch)
                        self.assertEqual(data_batch_['text'][0][0], str(i+1))
                        i = (i+1) % 1000
                    except tf.errors.OutOfRangeError:
                        print('Train data limit reached')
                        self.assertEqual(i, 0)
                        break

                iterator.switch_to_test_data(sess)
                i = 1001
                while True:
                    try:
                        data_batch_ = sess.run(data_batch)
                        self.assertEqual(data_batch_['text'][0][0], str(i))
                        i += 1
                    except tf.errors.OutOfRangeError:
                        print('Test data limit reached')
                        self.assertEqual(i, 2001)
                        break

if __name__ == "__main__":
    tf.test.main()
