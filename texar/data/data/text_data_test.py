# -*- coding: utf-8 -*-
#
"""
Unit tests for data related operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile
import copy

import tensorflow as tf

import texar as tx
#from texar.utils.exceptions import TexarError

# pylint: disable=too-many-locals

class MonoTextDataTest(tf.test.TestCase):
    """Tests text data class.
    """

    def setUp(self):
        tf.test.TestCase.setUp(self)

        # Create test data
        vocab_list = ['word', '词']
        vocab_file = tempfile.NamedTemporaryFile()
        vocab_file.write('\n'.join(vocab_list).encode("utf-8"))
        vocab_file.flush()
        self._vocab_file = vocab_file
        self._vocab_size = len(vocab_list)

        text = ['This is a test sentence .', '词 词 。']
        text_file = tempfile.NamedTemporaryFile()
        text_file.write('\n'.join(text).encode("utf-8"))
        text_file.flush()
        self._text_file = text_file

        self._hparams = {
            "num_epochs": 50,
            "batch_size": 3,
            "dataset": {
                "files": self._text_file.name,
                "vocab_file": self._vocab_file.name,
            }
        }

    def _run_and_test(self, hparams, test_batch_size=False):
        # Construct database
        text_data = tx.data.MonoTextData(hparams)
        self.assertEqual(text_data.vocab.vocab_size,
                         2 + len(text_data.vocab.special_tokens))

        iterator = text_data.dataset.make_initializable_iterator()
        text_data_batch = iterator.get_next()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            sess.run(iterator.initializer)

            while True:
                try:
                    # Run the logics
                    data_batch_ = sess.run(text_data_batch)
                    self.assertEqual(set(data_batch_.keys()),
                                     set(text_data.list_items()))
                    if test_batch_size:
                        self.assertEqual(len(data_batch_['text']),
                                         hparams['batch_size'])

                except tf.errors.OutOfRangeError:
                    print('Done -- epoch limit reached')
                    break

    def test_default_setting(self):
        """Tests the logics of MonoTextData.
        """
        self._run_and_test(self._hparams)

    def test_batching(self):
        """Tests different batching.
        """
        # dis-allow smaller final batch
        hparams = copy.copy(self._hparams)
        hparams.update({"allow_smaller_final_batch": False})
        self._run_and_test(hparams, test_batch_size=True)

        ## bucketing
        #raise TexarError("Only usible on TF-1.7")
        #hparams = copy.copy(self._hparams)
        #hparams.update({
        #    "bucket_boundaries": [2, 4, 6],
        #    "bucket_batch_sizes": [6, 4, 2]})
        #self._run_and_test(hparams, test_batch_size=True)

    def test_shuffle(self):
        """Tests different shuffle strategies.
        """
        hparams = copy.copy(self._hparams)
        hparams.update({
            "shard_and_shuffle": True,
            "shuffle_buffer_size": 1})
        self._run_and_test(hparams)

    def test_prefetch(self):
        """Tests prefetching.
        """
        hparams = copy.copy(self._hparams)
        hparams.update({"prefetch_buffer_size": 2})
        self._run_and_test(hparams)


    def test_other_transformations(self):
        """Tests use of other transformations
        """
        def _transform(x, data): # pylint: disable=invalid-name
            x[data.length_tensor_name] += 1
            return x

        hparams = copy.copy(self._hparams)
        hparams["dataset"].update({"other_transformations": [_transform]})
        self._run_and_test(hparams)


if __name__ == "__main__":
    tf.test.main()
