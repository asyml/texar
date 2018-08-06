# -*- coding: utf-8 -*-
#
"""
Unit tests for executor.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile
import shutil

import tensorflow as tf

from texar.run.executor import Executor
from texar.models.seq2seq.basic_seq2seq import BasicSeq2seq

class ExecutorTest(tf.test.TestCase):
    """Tests :class:`texar.run.executor.Executor`
    """

    def setUp(self):
        tf.test.TestCase.setUp(self)

        # Create data
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

        self._data_hparams = {
            "num_epochs": 20,
            "batch_size": 2,
            "source_dataset": {
                "files": [self._src_text_file.name],
                "vocab_file": self._vocab_file.name,
            },
            "target_dataset": {
                "files": self._tgt_text_file.name,
                "vocab_share": True,
            }
        }

    def test_execute_seq2seq(self):
        """Tests running seq2seq with Executor.
        """
        seq2seq = BasicSeq2seq(self._data_hparams)
        data_hparams = {'train': self._data_hparams, 'eval': self._data_hparams}

        model_dir = tempfile.mkdtemp()
        config = tf.estimator.RunConfig(
            model_dir=model_dir,
            save_summary_steps=10,
            save_checkpoints_steps=10,
            save_checkpoints_secs=None)

        exor = Executor(model=seq2seq, data_hparams=data_hparams, config=config)

        exor.train_and_evaluate(max_train_steps=20, eval_steps=5)

        exor.train(max_steps=20)
        exor.evaluate(steps=5)

        shutil.rmtree(model_dir)

if __name__ == "__main__":
    tf.test.main()
