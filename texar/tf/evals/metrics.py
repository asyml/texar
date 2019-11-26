"""
Various metrics.
"""

import tensorflow as tf

__all__ = [
    "accuracy",
    "binary_clas_accuracy"
]


def accuracy(labels, preds):
    """Calculates the accuracy of predictions.

    Args:
        labels: The ground truth values. A Tensor of the same shape of
            :attr:`preds`.
        preds: A Tensor of any shape containing the predicted values.

    Returns:
        A float scalar Tensor containing the accuracy.
    """
    labels = tf.cast(labels, preds.dtype)
    return tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))


def binary_clas_accuracy(pos_preds=None, neg_preds=None):
    """Calculates the accuracy of binary predictions.

    Args:
        pos_preds (optional): A Tensor of any shape containing the
            predicted values on positive data (i.e., ground truth labels are
            `1`).
        neg_preds (optional): A Tensor of any shape containing the
            predicted values on negative data (i.e., ground truth labels are
            `0`).

    Returns:
        A float scalar Tensor containing the accuracy.
    """
    pos_accu = accuracy(tf.ones_like(pos_preds), pos_preds)
    neg_accu = accuracy(tf.zeros_like(neg_preds), neg_preds)
    psize = tf.cast(tf.size(pos_preds), tf.float32)
    nsize = tf.cast(tf.size(neg_preds), tf.float32)
    accu = (pos_accu * psize + neg_accu * nsize) / (psize + nsize)
    return accu
