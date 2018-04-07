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
from texar.data import SpecialTokens

# pylint: disable=too-many-locals, too-many-branches, protected-access
# pylint: disable=invalid-name

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

    def _run_and_test(self, hparams, proc_shr=False, length_inc=None,
                      discard_src=False):
        # Construct database
        text_data = tx.data.PairedTextData(hparams)
        self.assertEqual(
            text_data.source_vocab.size,
            self._vocab_size + len(text_data.source_vocab.special_tokens))

        iterator = text_data.dataset.make_initializable_iterator()
        text_data_batch = iterator.get_next()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            sess.run(iterator.initializer)

            if proc_shr:
                tgt_eos = b'<EOS>'
            else:
                tgt_eos = b'<TARGET_EOS>'

            while True:
                try:
                    # Run the logics
                    data_batch_ = sess.run(text_data_batch)
                    self.assertEqual(set(data_batch_.keys()),
                                     set(text_data.list_items()))
                    # Test matching
                    src_text = data_batch_['source_text']
                    tgt_text = data_batch_['target_text']
                    if proc_shr:
                        for src, tgt in zip(src_text, tgt_text):
                            np.testing.assert_array_equal(src[:3], tgt[:3])
                    else:
                        for src, tgt in zip(src_text, tgt_text):
                            np.testing.assert_array_equal(src[:3], tgt[1:4])
                    self.assertTrue(
                        tgt_eos in data_batch_['target_text'][0])

                    if length_inc:
                        for i in range(len(data_batch_['source_text'])):
                            text_ = data_batch_['source_text'][i].tolist()
                            self.assertEqual(
                                text_.index(b'<EOS>') + 1,
                                data_batch_['source_length'][i] - length_inc[0])
                        for i in range(len(data_batch_['target_text'])):
                            text_ = data_batch_['target_text'][i].tolist()
                            self.assertEqual(
                                text_.index(tgt_eos) + 1,
                                data_batch_['target_length'][i] - length_inc[1])

                    if discard_src:
                        src_hparams = text_data.hparams.source_dataset
                        max_l = src_hparams.max_seq_length
                        max_l += text_data._decoder[0].added_length
                        for l in data_batch_[text_data.source_length_name]:
                            self.assertLessEqual(l, max_l)

                except tf.errors.OutOfRangeError:
                    print('Done -- epoch limit reached')
                    break

    def test_default_setting(self):
        """Tests the logics of the text data.
        """
        self._run_and_test(self._hparams)

    def test_shuffle(self):
        """Tests toggling shuffle.
        """
        hparams = copy.copy(self._hparams)
        hparams["shuffle"] = False
        self._run_and_test(hparams)

    def test_processing_share(self):
        """Tests sharing processing.
        """
        hparams = copy.copy(self._hparams)
        hparams["target_dataset"]["processing_share"] = True
        self._run_and_test(hparams, proc_shr=True)

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

    def test_length_filter(self):
        """Tests filtering by length.
        """
        hparams = copy.copy(self._hparams)
        hparams["source_dataset"].update(
            {"max_seq_length": 4,
             "length_filter_mode": "discard"})
        self._run_and_test(hparams, discard_src=True)

    #def test_sequence_length(self):
    #    hparams = {
    #        "batch_size": 64,
    #        "num_epochs": 1,
    #        "shuffle": False,
    #        "allow_smaller_final_batch": False,
    #        "source_dataset": {
    #            "files": "../../../data/yelp/sentiment.dev.sort.0",
    #            "vocab_file": "../../../data/yelp/vocab",
    #            "bos_token": SpecialTokens.BOS,
    #            "eos_token": SpecialTokens.EOS,
    #        },
    #        "target_dataset": {
    #            "files": "../../../data/yelp/sentiment.dev.sort.1",
    #            "vocab_share": True,
    #        },
    #    }
    #    data = tx.data.PairedTextData(hparams)

    #    iterator = tx.data.TrainTestDataIterator(val=data)
    #    text_data_batch = iterator.get_next()

    #    with self.test_session() as sess:
    #        sess.run(tf.global_variables_initializer())
    #        sess.run(tf.local_variables_initializer())
    #        sess.run(tf.tables_initializer())
    #        iterator.switch_to_val_data(sess)

    #        while True:
    #            try:
    #                data_batch_ = sess.run(text_data_batch)
    #                src = data_batch_["source_text_ids"]
    #                src_len = data_batch_["source_length"]
    #                self.assertEqual(src.shape[1], np.max(src_len))
    #                tgt = data_batch_["target_text_ids"]
    #                tgt_len = data_batch_["target_length"]
    #                self.assertEqual(tgt.shape[1], np.max(tgt_len))
    #            except tf.errors.OutOfRangeError:
    #                break

if __name__ == "__main__":
    tf.test.main()
