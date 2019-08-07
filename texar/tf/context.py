# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
    "global_mode_predict",
    "valid_modes"
]

_GLOBAL_MODE_KEY = "GLOBAL_MODE"

def global_mode():
    """Returns the Tensor of global mode.

    This is a placeholder with default value of
    :tf_main:`tf.estimator.ModeKeys.TRAIN <estimator/ModeKeys>`.

    Example:

        .. code-block:: python

            mode = session.run(global_mode())
            # mode == tf.estimator.ModeKeys.TRAIN

            mode = session.run(
                global_mode(),
                feed_dict={tf.global_mode(): tf.estimator.ModeKeys.PREDICT})
            # mode == tf.estimator.ModeKeys.PREDICT
    """
    mode = tf.get_collection_ref(_GLOBAL_MODE_KEY)
    if len(mode) < 1:
        #mode_tensor = tf.placeholder(tf.string, name="global_mode")
        mode_tensor = tf.placeholder_with_default(
            input=tf.estimator.ModeKeys.TRAIN,
            shape=(),
            name="global_mode")
        #mode_tensor = tf.constant(
        #    value=tf.estimator.ModeKeys.TRAIN,
        #    dtype=tf.string,
        #    name="global_mode")
        mode.append(mode_tensor)
    return mode[0]

def global_mode_train():
    """Returns a bool Tensor indicating whether the global mode is TRAIN.

    Example:

        .. code-block:: python

            is_train = session.run(global_mode_train())
            # is_train == True

            is_train = session.run(
                global_mode_train()
                feed_dict={tf.global_mode(): tf.estimator.ModeKeys.PREDICT})
            # is_train == False
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

def valid_modes():
    """Returns a set of possible values of mode.
    """
    return {tf.estimator.ModeKeys.TRAIN,
            tf.estimator.ModeKeys.EVAL,
            tf.estimator.ModeKeys.PREDICT}
