# -*- coding: utf-8 -*-
#
"""
Unit tests for database related operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile
import numpy as np

import tensorflow as tf

import texar.data

# pylint: disable=too-many-locals

class TextDataBaseTest(tf.test.TestCase):
    """Tests text database class.
    """

    def test_mono_text_database(self):
        """Tests the logics of MonoTextDataBase.
        """
        # Create test data
        vocab_list = ['word', '词']
        vocab_file = tempfile.NamedTemporaryFile()
        vocab_file.write('\n'.join(vocab_list).encode("utf-8"))
        vocab_file.flush()

        text = ['This is a test sentence .', '词 词 。']
        text_file = tempfile.NamedTemporaryFile()
        text_file.write('\n'.join(text).encode("utf-8"))
        text_file.flush()

        # Construct database
        hparams = {
            "num_epochs": 5,
            "batch_size": 3,
            "dataset": {
                "files": [text_file.name],
                "vocab_file": vocab_file.name,
            }
        }

        text_database = texar.data.MonoTextDataBase(hparams)
        text_data_batch = text_database()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                while not coord.should_stop():
                    # Run the logics
                    data = sess.run(text_data_batch)
                    self.assertEqual(set(data.keys()),
                                     set(text_database.list_items()))
                    self.assertEqual(len(data['text']), hparams['batch_size'])
                    self.assertEqual(text_database.vocab.vocab_size,
                                     len(vocab_list) + 4)

            except tf.errors.OutOfRangeError:
                print('Done -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)

    def test_paired_text_database(self):
        """Tests the logics of PairedTextDataBase.
        """
        # Create test data
        vocab_list = ['This', 'is', 'a', 'word', '词']
        vocab_file = tempfile.NamedTemporaryFile()
        vocab_file.write('\n'.join(vocab_list).encode("utf-8"))
        vocab_file.flush()

        src_text = ['This is a sentence from source .', '词 词 。 source']
        src_text_file = tempfile.NamedTemporaryFile()
        src_text_file.write('\n'.join(src_text).encode("utf-8"))
        src_text_file.flush()

        tgt_text = ['This is a sentence from target .', '词 词 。 target']
        tgt_text_file = tempfile.NamedTemporaryFile()
        tgt_text_file.write('\n'.join(tgt_text).encode("utf-8"))
        tgt_text_file.flush()

        # Construct database
        hparams = {
            "num_epochs": 100,
            "batch_size": 3,
            "source_dataset": {
                "files": [src_text_file.name],
                "vocab_file": vocab_file.name,
            },
            "target_dataset": {
                "files": [tgt_text_file.name],
                "vocab_share": True,
                "reader_share": True,
                "processing": {
                    "eos_token": "<TARGET_EOS>"
                }
            }
        }

        text_database = texar.data.PairedTextDataBase(hparams)
        text_data_batch = text_database()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                while not coord.should_stop():
                    # Run the logics
                    data = sess.run(text_data_batch)

                    self.assertEqual(set(data.keys()),
                                     set(text_database.list_items()))
                    self.assertEqual(len(data['source_text']),
                                     hparams['batch_size'])
                    self.assertEqual(len(data['target_text']),
                                     hparams['batch_size'])
                    self.assertEqual(text_database.source_vocab.vocab_size,
                                     len(vocab_list) + 4)
                    self.assertEqual(text_database.target_vocab.vocab_size,
                                     len(vocab_list) + 4)

                    src_text = data['source_text']
                    tgt_text = data['target_text']
                    for src, tgt in zip(src_text, tgt_text):
                        np.testing.assert_array_equal(
                            src[:3], tgt[1:4])

            except tf.errors.OutOfRangeError:
                print('Done -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)

    def test_multi_aligned_database(self):
        """Tests the logics of MultiAlignedDataBase.
        """
        # Create test data
        vocab_list = ['This', 'is', 'a', 'word', '词']
        vocab_file = tempfile.NamedTemporaryFile()
        vocab_file.write('\n'.join(vocab_list).encode("utf-8"))
        vocab_file.flush()

        text_0 = ['This is a sentence from source .', '词 词 。 source']
        text_0_file = tempfile.NamedTemporaryFile()
        text_0_file.write('\n'.join(text_0).encode("utf-8"))
        text_0_file.flush()

        text_1 = ['This is a sentence from target .', '词 词 。 target']
        text_1_file = tempfile.NamedTemporaryFile()
        text_1_file.write('\n'.join(text_1).encode("utf-8"))
        text_1_file.flush()

        int_2 = [0, 1]
        int_2_file = tempfile.NamedTemporaryFile()
        int_2_file.write('\n'.join([str(_) for _ in int_2]))
        int_2_file.flush()

        # Construct database
        hparams = {
            "num_epochs": 123,
            "batch_size": 64,
            "datasets": [
                { # dataset 0
                    "files": [text_0_file.name],
                    "vocab_file": vocab_file.name,
                },
                { # dataset 1
                    "files": [text_1_file.name],
                    "vocab_share_with": 0,
                    "reader_share_with": 0,
                    "processing": {
                        "eos_token": "<TARGET_EOS>"
                    }
                },
                { # dataset 2
                    "files": [int_2_file.name],
                    "data_type": "int",
                    "data_name_prefix": "label"
                }
            ]
        }

        database = texar.data.MultiAlignedDataBase(hparams)
        data_batch = database()

        self.assertEqual(len(database.datasets), len(hparams["datasets"]))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                while not coord.should_stop():
                    # Run the logics
                    data = sess.run(data_batch)

                    self.assertEqual(set(data.keys()),
                                     set(database.list_items()))
                    self.assertEqual(len(data['0_text']),
                                     hparams['batch_size'])
                    self.assertEqual(len(data['1_text']),
                                     hparams['batch_size'])
                    self.assertEqual(len(data['2_label']),
                                     hparams['batch_size'])
                    self.assertEqual(database.datasets[0].vocab.vocab_size,
                                     len(vocab_list) + 4)
                    self.assertEqual(database.datasets[1].vocab.vocab_size,
                                     len(vocab_list) + 4)

                    text_0 = data['0_text']
                    text_1 = data['1_text']
                    int_2 = data['2_label']
                    for t_0, t_1, i_2 in zip(text_0, text_1, int_2):
                        np.testing.assert_array_equal(
                            t_0[:4], t_1[:4])
                        if t_0[1].startswith(b'This'):
                            self.assertEqual(i_2, 0)
                        else:
                            self.assertEqual(i_2, 1)

            except tf.errors.OutOfRangeError:
                print('Done -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)


    def test_multi_source_text_database(self):
        """Tests the logics of MultiSourceTextDataBase.
        """
        # Create test data
        vocab_list = ['word', 'sentence', '词', 'response', 'dialog', '1', '2']
        vocab_file = tempfile.NamedTemporaryFile()
        vocab_file.write('\n'.join(vocab_list).encode("utf-8"))
        vocab_file.flush()

        source_text = [
            'This is a dialog 1 sentence . ||| This is a dialog 1 sentence . '
            '||| This is yet another dialog 1 sentence .',
            'This is a dialog 2 sentence . ||| '
            'This is also a dialog 2 sentence . '
        ]
        src_text_file = tempfile.NamedTemporaryFile()
        src_text_file.write('\n'.join(source_text).encode("utf-8"))
        src_text_file.flush()

        target_text = [
            'dialog 1 response', 'dialog 2 response 词'
        ]
        tgt_text_file = tempfile.NamedTemporaryFile()
        tgt_text_file.write('\n'.join(target_text).encode("utf-8"))
        tgt_text_file.flush()

        # Construct database
        hparams = {
            "num_epochs": 3,
            "batch_size": 3,
            "source_dataset": {
                "files": [src_text_file.name],
                "vocab_file": vocab_file.name,
                "processing": {
                    "max_seq_length": 10,
                    "max_context_length": 3
                }
            },
            "target_dataset": {
                "files": [tgt_text_file.name],
                "vocab_share": True,
                "reader_share": True,
                "processing": {
                    "eos_token": "<TARGET_EOS>"
                }
            }
        }

        text_database = texar.data.MultiSourceTextDataBase(hparams)
        text_data_batch = text_database()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                while not coord.should_stop():
                    # Run the logics
                    data = sess.run(text_data_batch)
                    self.assertEqual(set(data.keys()),
                                     set(text_database.list_items()))
                    self.assertEqual(len(data['source_text']),
                                     hparams['batch_size'])
                    self.assertEqual(len(data['target_text']),
                                     hparams['batch_size'])
                    self.assertEqual(text_database.source_vocab.vocab_size,
                                     len(vocab_list) + 4)
                    self.assertEqual(text_database.target_vocab.vocab_size,
                                     len(vocab_list) + 4)

            except tf.errors.OutOfRangeError:
                print('Done -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    tf.test.main()
