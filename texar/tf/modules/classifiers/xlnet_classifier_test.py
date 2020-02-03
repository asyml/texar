#
"""
Unit tests for XLNet classifier.
"""

import numpy as np
import tensorflow as tf

from texar.tf.modules.classifiers.xlnet_classifier import XLNetClassifier
from texar.tf.utils.test import pretrained_test

# pylint: disable=too-many-locals, no-member


class XLNetClassifierTest(tf.test.TestCase):
    """Tests :class:`~texar.tf.modules.XLNetClassifier` class.
    """

    @pretrained_test
    def test_model_loading(self):
        r"""Tests model loading functionality."""

        inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])

        for pretrained_model_name in XLNetClassifier.available_checkpoints():
            classifier = XLNetClassifier(
                pretrained_model_name=pretrained_model_name)
            _, _ = classifier(inputs)

    def test_trainable_variables(self):
        """Tests the functionality of automatically collecting trainable
        variables.
        """
        inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])

        # case 1
        hparams = {
            "pretrained_model_name": None,
        }
        clas = XLNetClassifier(hparams=hparams)
        clas(inputs)
        n_xlnet_vars = 162
        n_projection_vars = 2
        n_logits_vars = 2
        self.assertEqual(len(clas.trainable_variables),
                         n_xlnet_vars + n_logits_vars + n_projection_vars)

        # case 2
        hparams = {
            "pretrained_model_name": None,
            "clas_strategy": "time_wise"
        }
        clas = XLNetClassifier(hparams=hparams)
        clas(inputs)
        self.assertEqual(len(clas.trainable_variables),
                         n_xlnet_vars + n_logits_vars + n_projection_vars)

        # case 3
        hparams = {
            "pretrained_model_name": None,
            "clas_strategy": "all_time"
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
        hparams = {
            "pretrained_model_name": None
        }
        clas = XLNetClassifier(hparams=hparams)
        logits, pred = clas(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run([logits, pred])
            self.assertEqual(logits_.shape, (batch_size,
                                             clas.hparams.num_classes))
            self.assertEqual(pred_.shape, (batch_size,))

        # case 2
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 10,
            "clas_strategy": "time_wise"
        }
        clas = XLNetClassifier(hparams=hparams)
        logits, pred = clas(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run([logits, pred])
            self.assertEqual(logits_.shape,
                             (batch_size, max_time, clas.hparams.num_classes))
            self.assertEqual(pred_.shape, (batch_size, max_time))

        # case 3
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 0,
            "clas_strategy": "time_wise"
        }
        clas = XLNetClassifier(hparams=hparams)
        logits, pred = clas(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run([logits, pred])
            self.assertEqual(logits_.shape,
                             (batch_size, max_time, clas.hparams.hidden_dim))
            self.assertEqual(pred_.shape, (batch_size, max_time))

        # case 4
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 3,
            "clas_strategy": "all_time",
            "use_projection": False,
            "vocab_size": 40000
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
            "pretrained_model_name": None,
            "num_classes": 1,
            "clas_strategy": "time_wise"
        }
        clas = XLNetClassifier(hparams=hparams)
        logits, pred = clas(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run([logits, pred])
            self.assertEqual(logits_.shape, (batch_size, max_time))
            self.assertEqual(pred_.shape, (batch_size, max_time))

        # case 2
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 1,
            "clas_strategy": "cls_time",
            "max_seq_len": max_time
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
            "pretrained_model_name": None,
            "num_classes": 1,
            "clas_strategy": "all_time",
            "max_seq_len": max_time
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
