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

    def _run_and_test(self, hparams, test_batch_size=False, length_inc=None):
        # Construct database
        text_data = tx.data.MonoTextData(hparams)
        self.assertEqual(text_data.vocab.vocab_size,
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
                    if test_batch_size:
                        self.assertEqual(len(data_batch_['text']),
                                         hparams['batch_size'])
                    if length_inc:
                        for i in range(len(data_batch_['text'])):
                            text_ = data_batch_['text'][i].tolist()
                            self.assertEqual(
                                text_.index(b'<EOS>') + 1,
                                data_batch_['length'][i] - length_inc)

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
                "max_utterance_cnt": 3,
                "max_seq_length": 10
            }
        }

    def _run_and_test(self, hparams):
        # Construct database
        text_data = tx.data.VarUttMonoTextData(hparams)
        self.assertEqual(text_data.vocab.vocab_size,
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

                except tf.errors.OutOfRangeError:
                    print('Done -- epoch limit reached')
                    break

    def test_default_setting(self):
        """Tests the logics of the text data.
        """
        self._run_and_test(self._hparams)


class PairedTextDataTest(tf.test.TestCase):
    """Tests paired text data class.
    """

    def setUp(self):
        tf.test.TestCase.setUp(self)

        # Create test data
        vocab_list = ['This', 'is', 'a', 'word', '词']
        vocab_file = tempfile.NamedTemporaryFile()
        vocab_file.write('\n'.join(vocab_list).encode("utf-8"))
        vocab_file.flush()
        self._vocab_file = vocab_file
        self._vocab_size = len(vocab_list)

        src_text = ['This is a sentence from source .', '词 词 。 source']
        src_text_file = tempfile.NamedTemporaryFile()
        src_text_file.write('\n'.join(src_text).encode("utf-8"))
        src_text_file.flush()
        self._src_text_file = src_text_file

        tgt_text = ['This is a sentence from target .', '词 词 。 target']
        tgt_text_file = tempfile.NamedTemporaryFile()
        tgt_text_file.write('\n'.join(tgt_text).encode("utf-8"))
        tgt_text_file.flush()
        self._tgt_text_file = tgt_text_file

        self._hparams = {
            "num_epochs": 50,
            "batch_size": 3,
            "source_dataset": {
                "files": [self._src_text_file.name],
                "vocab_file": self._vocab_file.name,
            },
            "target_dataset": {
                "files": self._tgt_text_file.name,
                "vocab_share": True,
                "eos_token": "<TARGET_EOS>"
            }
        }

    def _run_and_test(self, hparams, length_inc=None):
        # Construct database
        text_data = tx.data.PairedTextData(hparams)
        self.assertEqual(
            text_data.source_vocab.vocab_size,
            self._vocab_size + len(text_data.source_vocab.special_tokens))

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
                    # Test matching
                    src_text = data_batch_['source_text']
                    tgt_text = data_batch_['target_text']
                    for src, tgt in zip(src_text, tgt_text):
                        np.testing.assert_array_equal(src[:3], tgt[1:4])

                    if length_inc:
                        for i in range(len(data_batch_['source_text'])):
                            text_ = data_batch_['source_text'][i].tolist()
                            self.assertEqual(
                                text_.index(b'<EOS>') + 1,
                                data_batch_['source_length'][i] - length_inc[0])
                        for i in range(len(data_batch_['target_text'])):
                            text_ = data_batch_['target_text'][i].tolist()
                            self.assertEqual(
                                text_.index(b'<TARGET_EOS>') + 1,
                                data_batch_['target_length'][i] - length_inc[1])

                except tf.errors.OutOfRangeError:
                    print('Done -- epoch limit reached')
                    break

    def test_default_setting(self):
        """Tests the logics of the text data.
        """
        self._run_and_test(self._hparams)

    def test_other_transformations(self):
        """Tests use of other transformations
        """
        def _transform(x, data_specs): # pylint: disable=invalid-name
            x[data_specs.decoder.length_tensor_name] += 1
            return x

        hparams = copy.copy(self._hparams)
        hparams["source_dataset"].update(
            {"other_transformations": [_transform, _transform]})
        hparams["target_dataset"].update(
            {"other_transformations": [_transform]})
        self._run_and_test(hparams, length_inc=(2, 1))


if __name__ == "__main__":
    tf.test.main()
