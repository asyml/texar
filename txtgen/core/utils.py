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
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import ops    # pylint: disable=E0611
from tensorflow.python.util import nest        # pylint: disable=E0611
from tensorflow.python.ops import rnn          # pylint: disable=E0611

from txtgen import context


MAX_SEQ_LENGTH = np.iinfo(np.int32).max # pylint: disable=no-member

def get_class(class_name, module_paths=None):
    """Returns the class based on class name.

    Args:
        class_name: Name or full path of the class to instantiate.
        module_paths: A list of paths to candidate modules to search for the
            class. This is used if the class cannot be located solely based on
            `class_name`. The first module in the list that contains the class
            is used.

    Returns:
        A class.

    Raises:
        ValueError: If class is not found based on :attr:`class_name` and
            :attr:`module_paths`.
    """
    class_ = locate(class_name)
    if (class_ is None) and (module_paths is not None):
        for module_path in module_paths:
            # Special treatment for module 'tensorflow.train' as
            # `import tensorflow.train` fails.
            if module_path == 'tensorflow.train':
                class_ = locate('.'.join([module_path, class_name]))
                if class_ is not None:
                    break
            module = importlib.import_module(module_path)
            if class_name in dir(module):
                class_ = getattr(module, class_name)
                break

    if class_ is None:
        raise ValueError(
            "Class not found in {}: {}".format(module_paths, class_name))

    return class_


def get_instance(class_name, kwargs, module_paths=None):
    """Creates a class instance.

    Args:
        class_name: Name or full path of the class to instantiate.
        kwargs: A dictionary of arguments for the class constructor.
        module_paths: A list of paths to candidate modules to search for the
            class. This is used if the class cannot be located solely based on
            `class_name`. The first module in the list that contains the class
            is used.

    Returns:
        A class instance.

    Raises:
        ValueError: If class is not found based on :attr:`class_name` and
            :attr:`module_paths`.
        ValueError: If :attr:`kwargs` contains arguments that are invalid
            for the class construction.
    """
    # Locate the class
    class_ = get_class(class_name, module_paths)

    # Check validity of arguments
    class_args = set(inspect.getargspec(class_.__init__).args) # pylint: disable=E1101
    for key in kwargs.keys():
        if key not in class_args:
            raise ValueError(
                "Invalid argument for class %s.%s: %s" %
                (class_.__module__, class_.__name__, key)) # pylint: disable=E1101

    return class_(**kwargs)


def get_instance_with_redundant_kwargs( # pylint: disable=invalid-name
        class_name, kwargs, module_paths=None):
    """Creates a class instance.

    Only those keyword arguments in :attr:`kwargs` that are included in the
    class construction method are used.

    Args:
        class_name (str): Name or full path of the class to instantiate.
        kwargs (dict): A dictionary of arguments for the class constructor. It
            may include invalid arguments which will be ignored.
        module_paths (list of str): A list of paths to candidate modules to
            search for the class. This is used if the class cannot be located
            solely based on :attr:`class_name`. The first module in the list
            that contains the class is used.

    Returns:
        A class instance.

    Raises:
        ValueError: If class is not found based on :attr:`class_name` and
            :attr:`module_paths`.
    """
    # Locate the class
    class_ = get_class(class_name, module_paths)

    # Select valid arguments
    selected_kwargs = {}
    class_args = set(inspect.getargspec(class_.__init__).args) # pylint: disable=E1101
    for key, value in kwargs.items():
        if key in class_args:
            selected_kwargs[key] = value

    return class_(**selected_kwargs)


def get_function(fn_name, module_paths=None):
    """Returns the function of specified name and module.

    Args:
        fn_name (str): Name of the function.
        module_paths (list of str): A list of paths to candidate modules to
            search for the function. This is used when the function cannot be
            located solely based on `fn_name`. The first module in the list
            that contains the function is used.

    Returns:
        A function.
    """
    fn = locate(fn_name)    # pylint: disable=invalid-name
    if (fn is None) and (module_paths is not None):
        for module_path in module_paths:
            # Special treatment for module 'tensorflow.train' as
            # `import tensorflow.train` fails.
            if module_path == 'tensorflow.train':
                fn = locate('.'.join([module_path, fn_name])) # pylint: disable=invalid-name
                if fn is not None:
                    break
            module = importlib.import_module(module_path)
            if fn_name in dir(module):
                fn = getattr(module, fn_name) # pylint: disable=invalid-name
                break

    if fn is None:
        raise ValueError(
            "Method not found in {}: {}".format(module_paths, fn_name))

    return fn


def call_function_with_redundant_kwargs(fn, kwargs):  # pylint: disable=invalid-name
    """Calls a function and returns the results.

    Only those keyword arguments in :attr:`kwargs` that are included in the
    function's argument list are used to call the function.

    Args:
        fn (function): The function to call.
        kwargs (dict): A dictionary of arguments for the class constructor. It
            may include invalid arguments which will be ignored.

    Returns:
        The returned results by calling :attr:`fn`.
    """
    # Select valid arguments
    selected_kwargs = {}
    fn_args = set(inspect.getargspec(fn).args)
    for key, value in kwargs.items():
        if key in fn_args:
            selected_kwargs[key] = value

    return fn(**selected_kwargs)


def get_default_arg_values(fn): # pylint: disable=invalid-name
    """Gets the arguments and respective default values of a function.

    Only arguments with default values are included in the output dictionary.

    Args:
        fn (function): The function to inspect.

    Returns:
        dict: A dictionary that maps argument names (str) to their default
        values. The dictionary is empty if no arguments have default values.
    """
    argspec = inspect.getargspec(fn)
    if argspec.defaults is None:
        return {}
    num_defaults = len(argspec.defaults)
    return dict(zip(argspec.args[-num_defaults:], argspec.defaults))


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
    # pylint: disable=protected-access
    flat_input = [rnn._transpose_batch_time(input_) for input_ in flat_input]
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


