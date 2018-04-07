#
"""
Global context manager that handles train/infer mode, etc
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

__all__ = [
    "global_mode",
    "global_mode_train",
    "global_mode_eval",
    "global_mode_predict"
]

_GLOBAL_MODE_KEY = "GLOBAL_MODE"

def global_mode():
    """Returns the placeholder of global mode.
    """
    mode = tf.get_collection_ref(_GLOBAL_MODE_KEY)
    if len(mode) < 1:
        mode.append(tf.placeholder(tf.string, name="global_mode"))
    return mode[0]

def global_mode_train():
    """Returns a bool Tensor indicating whether the global mode is TRAIN.
    """
    mode = global_mode()
    return tf.equal(mode, tf.estimator.ModeKeys.TRAIN)

def global_mode_eval():
    """Returns a bool Tensor indicating whether the global mode is EVAL.
    """
    mode = global_mode()
    return tf.equal(mode, tf.estimator.ModeKeys.EVAL)

def global_mode_predict():
    """Returns a bool Tensor indicating whether the global mode is PREDICT.
    """
    mode = global_mode()
    return tf.equal(mode, tf.estimator.ModeKeys.PREDICT)

