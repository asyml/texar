"""
Various metrics.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import tensorflow as tf

__all__ = [
    "accuarcy",
    "binary_clas_accuracy"
]

def accuarcy(predictions, labels):
    """Calculates the accuracy of predictions.

    Args:
        predictions: A Tensor of any shape containing the predicted values.
        labels: The ground truth values. A Tensor of the same shape of
            :attr:`predictions`.

    Returns:
        A float scalar Tensor containing the accuracy.
    """
    labels = tf.cast(labels, predictions.dtype)
    return tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))

def binary_clas_accuracy(pos_predictions=None, neg_predictions=None):
    """Calculates the accuracy of binary predictions.

    Args:
        pos_predictions (optional): A Tensor of any shape containing the
            predicted values on positive data (i.e., ground truth labels are
            `1`).
        neg_predictions (optional): A Tensor of any shape containing the
            predicted values on negative data (i.e., ground truth labels are
            `0`).

    Returns:
        A float scalar Tensor containing the accuracy.
    """
    pos_accu = accuarcy(pos_predictions, tf.ones_like(pos_predictions))
    neg_accu = accuarcy(neg_predictions, tf.zeros_like(neg_predictions))
    psize = tf.to_float(tf.size(pos_predictions))
    nsize = tf.to_float(tf.size(neg_predictions))
    accu = (pos_accu * psize + neg_accu * nsize) / (psize + nsize)
    return accu
