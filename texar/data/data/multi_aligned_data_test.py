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

# pylint: disable=too-many-locals, too-many-branches, protected-access

class MultiAlignedDataTest(tf.test.TestCase):
    """Tests multi aligned text data class.
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

        text_0 = ['This is a sentence from source .', '词 词 。 source']
        text_0_file = tempfile.NamedTemporaryFile()
        text_0_file.write('\n'.join(text_0).encode("utf-8"))
        text_0_file.flush()
        self._text_0_file = text_0_file

        text_1 = ['This is a sentence from target .', '词 词 。 target']
        text_1_file = tempfile.NamedTemporaryFile()
        text_1_file.write('\n'.join(text_1).encode("utf-8"))
        text_1_file.flush()
        self._text_1_file = text_1_file

        text_2 = [
            'This is a sentence from dialog . ||| dialog ',
            '词 词 。 ||| 词 dialog']
        text_2_file = tempfile.NamedTemporaryFile()
        text_2_file.write('\n'.join(text_2).encode("utf-8"))
        text_2_file.flush()
        self._text_2_file = text_2_file

        int_3 = [0, 1]
        int_3_file = tempfile.NamedTemporaryFile()
        int_3_file.write(('\n'.join([str(_) for _ in int_3])).encode("utf-8"))
        int_3_file.flush()
        self._int_3_file = int_3_file

        # Construct database
        self._hparams = {
            "num_epochs": 123,
            "batch_size": 23,
            "datasets": [
                { # dataset 0
                    "files": [self._text_0_file.name],
                    "vocab_file": self._vocab_file.name,
                    "bos_token": "",
                    "data_name": "0"
                },
                { # dataset 1
                    "files": [self._text_1_file.name],
                    "vocab_share_with": 0,
                    "eos_token": "<TARGET_EOS>",
                    "data_name": "1"
                },
                { # dataset 2
                    "files": [self._text_2_file.name],
                    "vocab_file": self._vocab_file.name,
                    "processing_share_with": 0,
                    "variable_utterance": True,
                    "data_name": "2"
                },
                { # dataset 3
                    "files": self._int_3_file.name,
                    "data_type": "int",
                    "data_name": "label"
                },
            ]
        }

    def _run_and_test(self, hparams, discard_did=None):
        # Construct database
        text_data = tx.data.MultiAlignedData(hparams)
        self.assertEqual(
            text_data.vocab(0).size,
            self._vocab_size + len(text_data.vocab(0).special_tokens))

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
                    self.assertEqual(text_data.utterance_cnt_name('2'),
                                     '2_utterance_cnt')
                    text_0 = data_batch_['0_text']
                    text_1 = data_batch_['1_text']
                    text_2 = data_batch_['2_text']
                    int_3 = data_batch_['label']
                    # pylint: disable=invalid-name
                    for t0, t1, t2, i3 in zip(text_0, text_1, text_2, int_3):
                        np.testing.assert_array_equal(
                            t0[:2], t1[1:3])
                        np.testing.assert_array_equal(
                            t0[:3], t2[0][:3])
                        if t0[0].startswith(b'This'):
                            self.assertEqual(i3, 0)
                        else:
                            self.assertEqual(i3, 1)

                    if discard_did is not None:
                        hpms = text_data._hparams.datasets[discard_did]
                        max_l = hpms.max_seq_length
                        max_l += text_data._decoder[discard_did].added_length
                        for i in range(2):
                            for length in data_batch_[text_data.length_name(i)]:
                                self.assertLessEqual(length, max_l)
                        for lengths in data_batch_[text_data.length_name(2)]:
                            for length in lengths:
                                self.assertLessEqual(length, max_l)
                    for i, hpms in enumerate(text_data._hparams.datasets):
                        if hpms.data_type != "text":
                            continue
                        max_l = hpms.max_seq_length
                        mode = hpms.length_filter_mode
                        if max_l is not None and mode == "truncate":
                            max_l += text_data._decoder[i].added_length
                            for length in data_batch_[text_data.length_name(i)]:
                                self.assertLessEqual(length, max_l)

                except tf.errors.OutOfRangeError:
                    print('Done -- epoch limit reached')
                    break

    def test_default_setting(self):
        """Tests the logics of the text data.
        """
        self._run_and_test(self._hparams)

    def test_length_filter(self):
        """Tests filtering by length.
        """
        hparams = copy.copy(self._hparams)
        hparams["datasets"][0].update(
            {"max_seq_length": 4,
             "length_filter_mode": "discard"})
        hparams["datasets"][1].update(
            {"max_seq_length": 2,
             "length_filter_mode": "truncate"})
        self._run_and_test(hparams, discard_did=0)



if __name__ == "__main__":
    tf.test.main()
