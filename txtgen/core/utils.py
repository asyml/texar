#
"""
Utility functions
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import inspect
import tensorflow as tf

from txtgen import context


def get_instance(class_module, args):
    """Creates an class instance

    Args:
      class_module: A class to instantiate
      args: A dictionary of arguments for the class constructor

    Returns:
      A class instance
    """
    # Check validity of params
    class_args = set(inspect.getargspec(class_module.__init__).args)
    for k in args.keys():
        if k not in class_args:
            raise ValueError(
                "Invalid argument for class %s: %s" % (class_module.__name__, k))

    return class_module(**args)


def switch_dropout(dropout_keep_prob):
    """Turn off dropout when not in training mode

    Args:
      dropout_keep_prob: dropout keep probability in training mode

    Returns:
      A unit Tensor that equals the dropout keep probability in training mode,
      and 1 in eval mode
    """
    return 1. - (1. - dropout_keep_prob) * tf.to_int32(context.is_train())
