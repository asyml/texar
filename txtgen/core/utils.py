#
"""
Utility functions
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import importlib
import inspect
from pydoc import locate

import tensorflow as tf
from tensorflow.python.framework import ops    # pylint: disable=E0611
from tensorflow.python.util import nest        # pylint: disable=E0611
from tensorflow.python.ops import rnn          # pylint: disable=E0611

from txtgen import context


def get_class(class_name, module_paths=None):
    """Returns the class based on class name.

    Args:
        class_name: Name (or full path) of the class to instantiate.
        module_paths: A list of paths to candidate modules to search for the
            class. This is used if the class cannot be located solely based on
            `class_name`. The first module in the list that contains the class
            is used.

    Returns:
        A class.
    """
    class_ = locate(class_name)
    if (class_ is None) and (module_paths is not None):
        for module_path in module_paths:
            module = importlib.import_module(module_path)
            if class_name in dir(module):
                class_ = getattr(module, class_name)
                break

    if class_ is None:
        raise ValueError(
            "Class not found in {}: {}".format(module_paths, class_name))

    return class_


def get_instance(class_name, kwargs, module_paths=None):
    """Creates an class instance.

    Args:
        class_name: Name (or full path) of the class to instantiate.
        kwargs: A dictionary of arguments for the class constructor.
        module_paths: A list of paths to candidate modules to search for the
            class. This is used if the class cannot be located solely based on
            `class_name`. The first module in the list that contains the class
            is used.

    Returns:
        A class instance.
    """
    # locate the class
    class_ = get_class(class_name, module_paths)

    # Check validity of params
    class_args = set(inspect.getargspec(class_.__init__).args)          # pylint: disable=E1101
    for k in kwargs.keys():
        if k not in class_args:
            raise ValueError("Invalid argument for class %s.%s: %s" %
                             (class_.__module__, class_.__name__, k))   # pylint: disable=E1101

    return class_(**kwargs)


def get_function(func_name, module_paths=None):
    """Returns the function of specified name and module.

    Args:
        func_name: Name of the function.
        module_paths: A list of paths to candidate modules to search for the
            function. This is used when the function cannot be located solely
            based on `func_name`. The first module in the list that contains the
            function is used.

    Returns:
        A function.
    """
    func = locate(func_name)
    if (func is None) and (module_paths is not None):
        for module_path in module_paths:
            module = importlib.import_module(module_path)
            if func_name in dir(module):
                func = getattr(module, func_name)
                break

    if func is None:
        raise ValueError(
            "Method not found in {}: {}".format(module_paths, func_name))

    return func


def switch_dropout(dropout_keep_prob, is_train=None):
    """Turns off dropout when not in training mode.

    Args:
        dropout_keep_prob: Dropout keep probability in training mode
        is_train: Boolean Tensor indicator of the training mode. Dropout is
            activated if `is_train=True`. If `is_train` is not given, the mode
            is inferred from the global mode.

    Returns:
        A unit Tensor that equals the dropout keep probability in training mode,
        and 1 in eval mode.
    """
    if is_train is None:
        return 1. - (1. - dropout_keep_prob) * tf.to_float(context.is_train())
    else:
        return 1. - (1. - dropout_keep_prob) * tf.to_float(is_train)


def transpose_batch_time(inputs):
    """Transposes inputs between time-major and batch-major.

    Args:
        inputs: A Tensor of shape `[batch_size, max_time, ...]` (batch-major)
            or `[max_time, batch_size, ...]` (time-major), or a (possibly
            nested) tuple of such elements.

    Returns:
        A Tensor with transposed batch and time dimensions of inputs.
    """
    flat_input = nest.flatten(inputs)
    flat_input = [ops.convert_to_tensor(input_) for input_ in flat_input]
    flat_input = [rnn._transpose_batch_time(input_) for input_ in flat_input]    # pylint: disable=protected-access
    return nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)


def default_string(str_, default_str):
    """Returns `str_` if `str_` is not None or empty, otherwise returns
    `default_str`.

    Args:
        str_: A string.
        default_str: A string.

    Returns:
        Either `str_` or `default_str`.
    """
    if str_ is not None or len(str_) > 0:
        return str_
    else:
        return default_str

