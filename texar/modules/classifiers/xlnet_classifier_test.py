#
"""
Unit tests for XLNet classifier.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf

from texar.modules.classifiers.xlnet_classifier import XLNetClassifier

# pylint: disable=too-many-locals, no-member


class XLNetClassifierTest(tf.test.TestCase):
    """Tests :class:`~texar.modules.XLNetClassifier` class.
    """

    def test_trainable_variables(self):
        """Tests the functionality of automatically collecting trainable
        variables.
        """
        inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])

        # case 1
        clas = XLNetClassifier()
        clas(inputs)
        n_xlnet_vars = 182
        n_projection_vars = 2
        n_logits_vars = 2
        self.assertEqual(len(clas.trainable_variables),
                         n_xlnet_vars + n_logits_vars + n_projection_vars)

        # case 2
        hparams = {
            "summary_type": "first"
        }
        clas = XLNetClassifier(hparams=hparams)
        clas(inputs)
        self.assertEqual(len(clas.trainable_variables),
                         n_xlnet_vars + n_logits_vars + n_projection_vars)

        # case 3
        hparams = {
            "summary_type": "mean",
        }
        clas = XLNetClassifier(hparams=hparams)
        clas(inputs)
        self.assertEqual(len(clas.trainable_variables),
                         n_xlnet_vars + n_logits_vars + n_projection_vars)

    def test_encode(self):
        """Tests encoding.
        """
        max_time = 8
        batch_size = 16
        inputs = tf.random_uniform([batch_size, max_time],
                                   maxval=30521, dtype=tf.int32)

        # case 1
        clas = XLNetClassifier()
        logits, pred = clas(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run([logits, pred])
            self.assertEqual(logits_.shape, (batch_size,
                                             clas.hparams.num_classes))
            self.assertEqual(pred_.shape, (batch_size,))

        # case 2
        hparams = {
            "num_classes": 10,
            "summary_type": "mean"
        }
        clas = XLNetClassifier(hparams=hparams)
        logits, pred = clas(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run([logits, pred])
            self.assertEqual(logits_.shape,
                             (batch_size, clas.hparams.num_classes))
            self.assertEqual(pred_.shape, (batch_size,))

        # case 3
        hparams = {
            "num_classes": 0,
            "summary_type": "first"
        }
        clas = XLNetClassifier(hparams=hparams)
        logits, pred = clas(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run([logits, pred])
            self.assertEqual(logits_.shape,
                             (batch_size, clas.hparams.hidden_dim))
            self.assertEqual(pred_.shape, (batch_size,))

        # case 4
        hparams = {
            "num_classes": 10,
            "summary_type": "mean"
        }
        inputs = tf.placeholder(tf.int32, shape=[batch_size, 6])
        clas = XLNetClassifier(hparams=hparams)
        logits, pred = clas(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run(
                [logits, pred],
                feed_dict={inputs: np.random.randint(30521,
                                                     size=(batch_size, 6))})
            self.assertEqual(logits_.shape, (batch_size,
                                             clas.hparams.num_classes))
            self.assertEqual(pred_.shape, (batch_size,))

    def test_binary(self):
        """Tests binary classification.
        """
        max_time = 8
        batch_size = 16
        inputs = tf.random_uniform([batch_size, max_time],
                                   maxval=30521, dtype=tf.int32)

        # case 1
        hparams = {
            "num_classes": 1,
            "summary_type": "first"
        }
        clas = XLNetClassifier(hparams=hparams)
        logits, pred = clas(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run([logits, pred])
            self.assertEqual(logits_.shape, (batch_size,))
            self.assertEqual(pred_.shape, (batch_size,))

        # case 2
        hparams = {
            "num_classes": 1,
            "summary_type": "last"
        }
        inputs = tf.placeholder(tf.int32, shape=[batch_size, 6])
        clas = XLNetClassifier(hparams=hparams)
        logits, pred = clas(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run(
                [logits, pred],
                feed_dict={inputs: np.random.randint(30521,
                                                     size=(batch_size, 6))})
            self.assertEqual(logits_.shape, (batch_size,))
            self.assertEqual(pred_.shape, (batch_size,))

        # case 3
        hparams = {
            "num_classes": 1,
            "summary_type": "mean"
        }
        inputs = tf.placeholder(tf.int32, shape=[batch_size, 6])
        clas = XLNetClassifier(hparams=hparams)
        logits, pred = clas(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run(
                [logits, pred],
                feed_dict={inputs: np.random.randint(30521,
                                                     size=(batch_size, 6))})
            self.assertEqual(logits_.shape, (batch_size,))
            self.assertEqual(pred_.shape, (batch_size,))


if __name__ == "__main__":
    tf.test.main()
