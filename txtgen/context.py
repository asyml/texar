#
"""
Global context manager that handles train/eval mode, etc
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


_IS_TRAIN_KEY = "CONTEXT_IS_TRAIN"

def is_train():
    """Returns the global mode indicator.

    Returns: A bool placeholder that indicates the global train/eval mode.
    """
    is_train_values = tf.get_collection_ref(_IS_TRAIN_KEY)
    if len(is_train_values) < 1:
        is_train_values.append(tf.placeholder(tf.bool, name="is_train"))
    return is_train_values[0]
