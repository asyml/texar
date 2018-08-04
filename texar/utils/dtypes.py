#
"""
Utility functions related to data types.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# pylint: disable=invalid-name, no-member

import six
import numpy as np

import tensorflow as tf

__all__ = [
    "get_tf_dtype",
    "is_callable",
    "is_str",
    "maybe_hparams_to_dict"
]

def get_tf_dtype(dtype): # pylint: disable=too-many-return-statements
    """Returns respective tf dtype.

    Args:
        dtype: A str, python numeric or string type, numpy data type, or
            tf dtype.

    Returns:
        The respective tf dtype.
    """
    if dtype in {'float', 'float32', 'tf.float32', float,
                 np.float32, tf.float32}:
        return tf.float32
    elif dtype in {'float64', 'tf.float64', np.float64, np.float_, tf.float64}:
        return tf.float64
    elif dtype in {'float16', 'tf.float16', np.float16, tf.float16}:
        return tf.float16
    elif dtype in {'int', 'int32', 'tf.int32', int, np.int32, tf.int32}:
        return tf.int32
    elif dtype in {'int64', 'tf.int64', np.int64, tf.int64}:
        return tf.int64
    elif dtype in {'int16', 'tf.int16', np.int16, tf.int16}:
        return tf.int16
    elif dtype in {'bool', 'tf.bool', bool, np.bool_, tf.bool}:
        return tf.bool
    elif dtype in {'string', 'str', 'tf.string', str, np.str, tf.string}:
        return tf.string
    try:
        if dtype == {'unicode', unicode}:
            return tf.string
    except NameError:
        pass

    raise ValueError(
        "Unsupported conversion from type {} to tf dtype".format(str(dtype)))

def is_callable(x):
    """Return `True` if :attr:`x` is callable.
    """
    try:
        _is_callable = callable(x)
    except: # pylint: disable=bare-except
        _is_callable = hasattr(x, '__call__')
    return _is_callable

def is_str(x):
    """Returns `True` if :attr:`x` is either a str or unicode. Returns `False`
    otherwise.
    """
    return isinstance(x, six.string_types)

def maybe_hparams_to_dict(hparams):
    """If :attr:`hparams` is an instance of :class:`~texar.hyperparams.HParams`,
    converts it to a `dict` and returns. If :attr:`hparams` is a `dict`,
    returns as is.
    """
    if isinstance(hparams, dict):
        return hparams
    return hparams.todict()

