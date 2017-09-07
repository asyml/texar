#
"""
Unit tests for various layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from txtgen import context
from txtgen.hyperparams import HParams
from txtgen.core import layers

class GetRNNCellTest(tf.test.TestCase):
    """Tests RNN cell creator.
    """

    def test_get_rnn_cell(self):
        """Tests get_rnn_cell.
        """
        emb_dim = 4
        num_units = 64
        hparams = {
            "cell": {
                "type": "tensorflow.contrib.rnn.GRUCell",
                "kwargs": {
                    "num_units": num_units
                }
            },
            "num_layers": 2,
            "dropout": {
                "input_keep_prob": 0.8,
                "variational_recurrent": True,
                "input_size": [emb_dim, num_units]
            },
            "residual": True,
            "highway": True
        }

        hparams_ = HParams(hparams, layers.default_rnn_cell_hparams())
        cell = layers.get_rnn_cell(hparams_)

        batch_size = 16
        inputs = tf.zeros([batch_size, emb_dim], dtype=tf.float32)
        output, state = cell(inputs,
                             cell.zero_state(batch_size, dtype=tf.float32))
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output_, state_ = sess.run([output, state],
                                       feed_dict={context.is_train(): True})
            self.assertEqual(output_.shape[0], batch_size)
            if isinstance(state_, (list, tuple)):
                self.assertEqual(state_[0].shape[0], batch_size)
                self.assertEqual(state_[0].shape[1],
                                 hparams_.cell.kwargs.num_units)
            else:
                self.assertEqual(state_.shape[0], batch_size)
                self.assertEqual(state_.shape[1],
                                 hparams_.cell.kwargs.num_units)


if __name__ == "__main__":
    tf.test.main()
