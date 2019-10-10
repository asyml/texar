"""
Unit tests for BERT classifiers.
"""

import numpy as np
import tensorflow as tf

from texar.tf.modules.classifiers.gpt2_classifier import GPT2Classifier
from texar.tf.utils.test import pretrained_test


class GPT2ClassifierTest(tf.test.TestCase):
    """Tests :class:`~texar.tf.modules.GPT2Classifier` class.
    """

    @pretrained_test
    def test_model_loading(self):
        r"""Tests model loading functionality."""

        inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])

        for pretrained_model_name in GPT2Classifier.available_checkpoints():
            classifier = GPT2Classifier(
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
        clas = GPT2Classifier(hparams=hparams)
        _, _ = clas(inputs)
        self.assertEqual(len(clas.trainable_variables), 198)

        # case 2
        hparams = {
            "pretrained_model_name": None,
            "clas_strategy": "all_time",
            "max_seq_length": 8,
        }
        clas = GPT2Classifier(hparams=hparams)
        _, _ = clas(inputs)
        self.assertEqual(len(clas.trainable_variables), 198)

        # case 3
        hparams = {
            "pretrained_model_name": None,
            "clas_strategy": "time_wise",
        }
        clas = GPT2Classifier(hparams=hparams)
        _, _ = clas(inputs)
        self.assertEqual(len(clas.trainable_variables), 198)

    def test_classification(self):
        r"""Tests classificaiton.
        """
        max_time = 8
        batch_size = 16
        inputs = tf.random_uniform([batch_size, max_time],
                                   maxval=30521, dtype=tf.int32)

        # case 1
        hparams = {
            "pretrained_model_name": None,
        }
        classifier = GPT2Classifier(hparams=hparams)
        logits, preds = classifier(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run([logits, preds])
            self.assertEqual(logits_.shape, (batch_size, 2))
            self.assertEqual(pred_.shape, (batch_size,))

        # case 2
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 10,
            "clas_strategy": "time_wise",
        }
        classifier = GPT2Classifier(hparams=hparams)
        logits, preds = classifier(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run([logits, preds])
            self.assertEqual(logits_.shape, (batch_size, max_time, 10))
            self.assertEqual(pred_.shape, (batch_size, max_time))

        # case 3
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 0,
            "clas_strategy": "time_wise",
        }
        classifier = GPT2Classifier(hparams=hparams)
        logits, preds = classifier(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run([logits, preds])
            self.assertEqual(logits_.shape, (batch_size, max_time, 768))
            self.assertEqual(pred_.shape, (batch_size, max_time))

        # case 4
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 10,
            "clas_strategy": "all_time",
            "max_seq_length": max_time,
        }
        classifier = GPT2Classifier(hparams=hparams)
        logits, preds = classifier(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run([logits, preds])
            self.assertEqual(logits_.shape, (batch_size, 10))
            self.assertEqual(pred_.shape, (batch_size,))

    def test_binary(self):
        r"""Tests binary classification.
        """
        max_time = 8
        batch_size = 16
        inputs = tf.random_uniform([batch_size, max_time],
                                   maxval=30521, dtype=tf.int32)

        # case 1
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 1,
            "clas_strategy": "time_wise",
        }
        classifier = GPT2Classifier(hparams=hparams)
        logits, preds = classifier(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run([logits, preds])
            self.assertEqual(logits_.shape, (batch_size, max_time))
            self.assertEqual(pred_.shape, (batch_size, max_time))

        # case 2
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 1,
            "clas_strategy": "cls_time",
            "max_seq_length": max_time,
        }
        classifier = GPT2Classifier(hparams=hparams)
        logits, preds = classifier(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run([logits, preds])
            self.assertEqual(logits_.shape, (batch_size,))
            self.assertEqual(pred_.shape, (batch_size,))

        # case 3
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 1,
            "clas_strategy": "all_time",
            "max_seq_length": max_time,
        }
        classifier = GPT2Classifier(hparams=hparams)
        logits, preds = classifier(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_, pred_ = sess.run([logits, preds])
            self.assertEqual(logits_.shape, (batch_size,))
            self.assertEqual(pred_.shape, (batch_size,))


if __name__ == "__main__":
    tf.test.main()
