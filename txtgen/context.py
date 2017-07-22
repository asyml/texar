#
"""
Global context manager that handles train/eval mode, etc
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


_IS_TRAIN = tf.placeholder(tf.bool, name="is_train")

is_train():
  return _IS_TRAIN
