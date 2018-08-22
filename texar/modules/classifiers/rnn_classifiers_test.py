#
"""
Unit tests for RNN classifiers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf

from texar.modules.classifiers.rnn_classifiers import \
        UnidirectionalRNNClassifier

# pylint: disable=too-many-locals, no-member

class UnidirectionalRNNClassifierTest(tf.test.TestCase):
    """Tests :class:`~texar.modules.UnidirectionalRNNClassifierTest` class.
    """

    def test_trainable_variables(self):
        """Tests the functionality of automatically collecting trainable
        variables.
        """
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, None, 100])

        # case 1
        clas = UnidirectionalRNNClassifier()
        _, _ = clas(inputs)
        self.assertEqual(len(clas.trainable_variables), 2+2)

        # case 2
        hparams = {
            "output_layer": {"num_layers": 2},
            "logit_layer_kwargs": {"use_bias": False}
        }
        clas = UnidirectionalRNNClassifier(hparams=hparams)
        _, _ = clas(inputs)
        self.assertEqual(len(clas.trainable_variables), 2+2+2+1)
        _, _ = clas(inputs)
        self.assertEqual(len(clas.trainable_variables), 2+2+2+1)

    def test_encode(self):
        """Tests encoding.
        """
        max_time = 8
        batch_size = 16
        emb_dim = 100
        inputs = tf.random_uniform([batch_size, max_time, emb_dim],
                                   maxval=1., dtype=tf.float32)

        # case 1
        clas = UnidirectionalRNNClassifier()
        logits, pred = clas(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run([logits, pred])
            self.assertEqual(logits_.shape, (batch_size, clas.num_classes))
            self.assertEqual(pred_.shape, (batch_size, ))

        # case 2
        hparams = {
            "num_classes": 10,
            "clas_strategy": "time_wise"
        }
        clas = UnidirectionalRNNClassifier(hparams=hparams)
        logits, pred = clas(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run([logits, pred])
            self.assertEqual(logits_.shape,
                             (batch_size, max_time, clas.num_classes))
            self.assertEqual(pred_.shape, (batch_size, max_time))

        # case 3
        hparams = {
            "output_layer": {
                "num_layers": 1,
                "layer_size": 10
            },
            "num_classes": 0,
            "clas_strategy": "time_wise"
        }
        clas = UnidirectionalRNNClassifier(hparams=hparams)
        logits, pred = clas(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run([logits, pred])
            self.assertEqual(logits_.shape,
                             (batch_size, max_time, 10))
            self.assertEqual(pred_.shape, (batch_size, max_time))


        # case 4
        hparams = {
            "num_classes": 10,
            "clas_strategy": "all_time",
            "max_seq_length": max_time
        }
        inputs = tf.placeholder(tf.float32, shape=[batch_size, 6, emb_dim])
        clas = UnidirectionalRNNClassifier(hparams=hparams)
        logits, pred = clas(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run(
                [logits, pred],
                feed_dict={inputs: np.random.randn(batch_size, 6, emb_dim)})
            self.assertEqual(logits_.shape, (batch_size, clas.num_classes))
            self.assertEqual(pred_.shape, (batch_size, ))

    def test_binary(self):
        """Tests binary classification.
        """
        max_time = 8
        batch_size = 16
        emb_dim = 100
        inputs = tf.random_uniform([batch_size, max_time, emb_dim],
                                   maxval=1., dtype=tf.float32)

        # case 1 omittd

        # case 2
        hparams = {
            "num_classes": 1,
            "clas_strategy": "time_wise"
        }
        clas = UnidirectionalRNNClassifier(hparams=hparams)
        logits, pred = clas(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run([logits, pred])
            self.assertEqual(logits_.shape, (batch_size, max_time))
            self.assertEqual(pred_.shape, (batch_size, max_time))

        # case 3
        hparams = {
            "output_layer": {
                "num_layers": 1,
                "layer_size": 10
            },
            "num_classes": 1,
            "clas_strategy": "time_wise"
        }
        clas = UnidirectionalRNNClassifier(hparams=hparams)
        logits, pred = clas(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run([logits, pred])
            self.assertEqual(logits_.shape, (batch_size, max_time))
            self.assertEqual(pred_.shape, (batch_size, max_time))


        # case 4
        hparams = {
            "num_classes": 1,
            "clas_strategy": "all_time",
            "max_seq_length": max_time
        }
        inputs = tf.placeholder(tf.float32, shape=[batch_size, 6, emb_dim])
        clas = UnidirectionalRNNClassifier(hparams=hparams)
        logits, pred = clas(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run(
                [logits, pred],
                feed_dict={inputs: np.random.randn(batch_size, 6, emb_dim)})
            self.assertEqual(logits_.shape, (batch_size, ))
            self.assertEqual(pred_.shape, (batch_size, ))

if __name__ == "__main__":
    tf.test.main()
