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
import numpy as np

import tensorflow as tf

import texar as tx

# pylint: disable=too-many-locals, protected-access, too-many-branches
# pylint: disable=invalid-name

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

    def _run_and_test(self,
                      hparams,
                      test_batch_size=False,
                      length_inc=None):
        # Construct database
        text_data = tx.data.MonoTextData(hparams)
        self.assertEqual(text_data.vocab.size,
                         self._vocab_size + len(text_data.vocab.special_tokens))

        iterator = text_data.dataset.make_initializable_iterator()
        text_data_batch = iterator.get_next()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            sess.run(iterator.initializer)

            while True:
                try:
                    data_batch_ = sess.run(text_data_batch)

                    self.assertEqual(set(data_batch_.keys()),
                                     set(text_data.list_items()))

                    if test_batch_size:
                        self.assertEqual(len(data_batch_['text']),
                                         hparams['batch_size'])

                    if length_inc:
                        for i in range(len(data_batch_['text'])):
                            text_ = data_batch_['text'][i].tolist()
                            self.assertEqual(
                                text_.index(b'<EOS>') + 1,
                                data_batch_['length'][i] - length_inc)

                    max_seq_length = text_data.hparams.dataset.max_seq_length
                    mode = text_data.hparams.dataset.length_filter_mode
                    if max_seq_length == 6:
                        max_l = max_seq_length
                        max_l += text_data._decoder.added_length
                        for length in data_batch_['length']:
                            self.assertLessEqual(length, max_l)
                        if mode == "discard":
                            for length in data_batch_['length']:
                                self.assertEqual(length, 5)
                        elif mode == "truncate":
                            num_length_6 = 0
                            for length in data_batch_['length']:
                                num_length_6 += int(length == 6)
                            self.assertGreater(num_length_6, 0)
                        else:
                            raise ValueError("Unknown mode: %s" % mode)

                    if text_data.hparams.dataset.pad_to_max_seq_length:
                        max_l = max_seq_length + text_data._decoder.added_length
                        for x in data_batch_['text']:
                            self.assertEqual(len(x), max_l)
                        for x in data_batch_['text_ids']:
                            self.assertEqual(len(x), max_l)

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

    def test_bucketing(self):
        """Tests bucketing.
        """
        hparams = copy.copy(self._hparams)
        hparams.update({
            "bucket_boundaries": [7],
            "bucket_batch_sizes": [6, 4]})

        text_data = tx.data.MonoTextData(hparams)
        iterator = text_data.dataset.make_initializable_iterator()
        text_data_batch = iterator.get_next()

        hparams.update({
            "bucket_boundaries": [7],
            "bucket_batch_sizes": [7, 7],
            "allow_smaller_final_batch": False})

        text_data_1 = tx.data.MonoTextData(hparams)
        iterator_1 = text_data_1.dataset.make_initializable_iterator()
        text_data_batch_1 = iterator_1.get_next()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            sess.run(iterator.initializer)
            sess.run(iterator_1.initializer)

            while True:
                try:
                    # Run the logics
                    data_batch_, data_batch_1_ = sess.run(
                        [text_data_batch, text_data_batch_1])

                    length_ = data_batch_['length'][0]
                    if length_ < 7:
                        last_batch_size = hparams['num_epochs'] % 6
                        self.assertTrue(
                            len(data_batch_['text']) == 6 or
                            len(data_batch_['text']) == last_batch_size)
                    else:
                        last_batch_size = hparams['num_epochs'] % 4
                        self.assertTrue(
                            len(data_batch_['text']) == 4 or
                            len(data_batch_['text']) == last_batch_size)

                    self.assertEqual(len(data_batch_1_['text']), 7)

                except tf.errors.OutOfRangeError:
                    print('Done -- epoch limit reached')
                    break

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
        def _transform(x, data_specs): # pylint: disable=invalid-name
            x[data_specs.decoder.length_tensor_name] += 1
            return x

        hparams = copy.copy(self._hparams)
        hparams["dataset"].update(
            {"other_transformations": [_transform, _transform]})
        self._run_and_test(hparams, length_inc=2)

    def test_list_items(self):
        """Tests the item names of the output data.
        """
        text_data = tx.data.MonoTextData(self._hparams)
        self.assertSetEqual(set(text_data.list_items()),
                            {"text", "text_ids", "length"})

        hparams = copy.copy(self._hparams)
        hparams["dataset"]["data_name"] = "data"
        text_data = tx.data.MonoTextData(hparams)
        self.assertSetEqual(set(text_data.list_items()),
                            {"data_text", "data_text_ids", "data_length"})

    def test_length_discard(self):
        """Tests discard lenghy seq.
        """
        hparams = copy.copy(self._hparams)
        hparams["dataset"].update({"max_seq_length": 4,
                                   "length_filter_mode": "discard"})
        self._run_and_test(hparams)

    def test_length_truncate(self):
        """Tests truncation.
        """
        hparams = copy.copy(self._hparams)
        hparams["dataset"].update({"max_seq_length": 4,
                                   "length_filter_mode": "truncate"})
        hparams["shuffle"] = False
        hparams["allow_smaller_final_batch"] = False
        self._run_and_test(hparams)

    def test_pad_to_max_length(self):
        """Tests padding.
        """
        hparams = copy.copy(self._hparams)
        hparams["dataset"].update({"max_seq_length": 10,
                                   "length_filter_mode": "truncate",
                                   "pad_to_max_seq_length": True})
        self._run_and_test(hparams)


class VarUttMonoTextDataTest(tf.test.TestCase):
    """Tests variable utterance text data class.
    """

    def setUp(self):
        tf.test.TestCase.setUp(self)

        # Create test data
        vocab_list = ['word', 'sentence', '词', 'response', 'dialog', '1', '2']
        vocab_file = tempfile.NamedTemporaryFile()
        vocab_file.write('\n'.join(vocab_list).encode("utf-8"))
        vocab_file.flush()
        self._vocab_file = vocab_file
        self._vocab_size = len(vocab_list)

        text = [
            'This is a dialog 1 sentence . ||| This is a dialog 1 sentence . '
            '||| This is yet another dialog 1 sentence .', #//
            'This is a dialog 2 sentence . ||| '
            'This is also a dialog 2 sentence . ', #//
            '词 词 词 ||| word', #//
            'This This', #//
            '1 1 1 ||| 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 ||| 1 1 1 ||| 2'
        ]
        text_file = tempfile.NamedTemporaryFile()
        text_file.write('\n'.join(text).encode("utf-8"))
        text_file.flush()
        self._text_file = text_file

        self._hparams = {
            "num_epochs": 50,
            "batch_size": 3,
            "shuffle": False,
            "dataset": {
                "files": self._text_file.name,
                "vocab_file": self._vocab_file.name,
                "variable_utterance": True,
                "max_utterance_cnt": 3,
                "max_seq_length": 10
            }
        }

    def _run_and_test(self, hparams):
        # Construct database
        text_data = tx.data.MonoTextData(hparams)
        self.assertEqual(text_data.vocab.size,
                         self._vocab_size + len(text_data.vocab.special_tokens))

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

                    # Test utterance count
                    utt_ind = np.sum(data_batch_["text_ids"], 2) != 0
                    utt_cnt = np.sum(utt_ind, 1)
                    self.assertListEqual(
                        data_batch_[text_data.utterance_cnt_name].tolist(),
                        utt_cnt.tolist())

                    if text_data.hparams.dataset.pad_to_max_seq_length:
                        max_l = text_data.hparams.dataset.max_seq_length
                        max_l += text_data._decoder.added_length
                        for x in data_batch_['text']:
                            for xx in x:
                                self.assertEqual(len(xx), max_l)
                        for x in data_batch_['text_ids']:
                            for xx in x:
                                self.assertEqual(len(xx), max_l)

                except tf.errors.OutOfRangeError:
                    print('Done -- epoch limit reached')
                    break

    def test_default_setting(self):
        """Tests the logics of the text data.
        """
        self._run_and_test(self._hparams)

    def test_pad_to_max_length(self):
        """Tests padding.
        """
        hparams = copy.copy(self._hparams)
        hparams["dataset"].update({"max_seq_length": 20,
                                   "length_filter_mode": "truncate",
                                   "pad_to_max_seq_length": True})
        self._run_and_test(hparams)

if __name__ == "__main__":
    tf.test.main()
