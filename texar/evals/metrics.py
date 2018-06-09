"""
Various metrics.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import tensorflow as tf

__all__ = [
    "accuarcy"
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
