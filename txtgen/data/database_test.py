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
import tensorflow as tf
#import numpy as np

from txtgen.data import database


class TextDatabaseTest(tf.test.TestCase):
    """Tests text database class.
    """

    def test_database_call(self):
        """Tests the database call function.
        """
        vocab_list = ['word', '词']
        vocab_file = tempfile.NamedTemporaryFile()
        vocab_file.write('\n'.join(vocab_list).encode("utf-8"))
        vocab_file.flush()

        text = ['This is a test sentence .', '词 词 。']
        text_file = tempfile.NamedTemporaryFile()
        text_file.write('\n'.join(text).encode("utf-8"))
        text_file.flush()

        hparams = {
            "num_epochs": 2,
            "batch_size": 2,
            "dataset": {
                "files": [text_file.name],
                "vocab.file": vocab_file.name,
            }
        }

        text_database = database.TextDataBase(hparams)
        text_data = text_database()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                while not coord.should_stop():
                    # Run the logics
                    data = sess.run(text_data)

                    self.assertEqual(set(data.keys()),
                                     set(text_database.list_items()))
                    self.assertEqual(len(data['text']), hparams['batch_size'])
                    self.assertEqual(text_database.dataset.vocab.vocab_size, # pylint: disable=no-member
                                     len(vocab_list) + 3)

            except tf.errors.OutOfRangeError:
                print('Done -- epoch limit reaached')
            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    tf.test.main()

