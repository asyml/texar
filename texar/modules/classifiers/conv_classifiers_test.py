#
"""
Unit tests for conv encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

import texar as tx
from texar.modules.classifiers.conv_classifiers import Conv1DClassifier


class Conv1DClassifierTest(tf.test.TestCase):
    """Tests :class:`~texar.modules.Conv1DClassifier` class.
    """

    def test_classifier(self):
        """Tests classification.
        """
        # case 1: default hparams
        classifier = Conv1DClassifier()
        self.assertEqual(len(classifier.layers), 5)
        self.assertTrue(isinstance(classifier.layers[-1],
                                   tf.layers.Dense))
        inputs = tf.ones([64, 16, 300], tf.float32)
        logits, pred = classifier(inputs)
        self.assertEqual(logits.shape, [64, 2])
        self.assertEqual(pred.shape, [64])

        inputs = tf.placeholder(tf.float32, [64, None, 300])
        logits, pred = classifier(inputs)
        self.assertEqual(logits.shape, [64, 2])
        self.assertEqual(pred.shape, [64])

        # case 1
        hparams = {
            "num_classes": 10,
            "logit_layer_kwargs": {"use_bias": False}
        }
        classifier = Conv1DClassifier(hparams=hparams)
        inputs = tf.ones([64, 16, 300], tf.float32)
        logits, pred = classifier(inputs)
        self.assertEqual(logits.shape, [64, 10])
        self.assertEqual(pred.shape, [64])


if __name__ == "__main__":
    tf.test.main()
