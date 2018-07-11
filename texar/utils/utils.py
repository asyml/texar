#
"""
Miscellaneous Utility functions.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# pylint: disable=invalid-name, no-member, no-name-in-module, protected-access

#import importlib
import inspect
from pydoc import locate
import copy
import numpy as np

import tensorflow as tf

from texar import context
from texar.hyperparams import HParams
from texar.utils.dtypes import is_str, is_callable

MAX_SEQ_LENGTH = np.iinfo(np.int32).max

## Some modules cannot be imported directly,
## e.g., `import tensorflow.train` fails.
## Such modules are treated in a special way in utils like `get_class` as below.
#_unimportable_modules = {
#    'tensorflow.train', 'tensorflow.keras.regularizers'
#}

__all__ = [
    "check_or_get_class",
    "get_class",
    "check_or_get_instance",
    "get_instance",
    "get_instance_with_redundant_kwargs",
    "get_function",
    "call_function_with_redundant_kwargs",
    "get_args",
    "get_default_arg_values",
    "get_instance_kwargs",
    "add_variable",
    "get_unique_named_variable_scope",
    "maybe_gloabl_mode",
    "is_train_mode",
    "is_eval_mode",
    "is_predict_mode",
    "is_train_mode_py",
    "is_eval_mode_py",
    "is_predict_mode_py",
    "switch_dropout",
    "default_string",
    "patch_dict",
    "fetch_subdict",
    "uniquify_str",
    "soft_sequence_embedding",
    "straight_through",
    "ceildiv",
]


# TODO(zhiting): complete this
def _expand_name(name):
    """Replaces common shorthands with respective full names.

        "tf.xxx" --> "tensorflow.xxx"
        "tx.xxx" --> "texar.xxx"
    """
    return name

def check_or_get_class(class_or_name, module_path=None, superclass=None):
    """Returns the class and checks if the class inherits :attr:`superclass`.

    Args:
        class_or_name: Name or full path to the class, or the class itself.
        module_paths (list, optional): Paths to candidate modules to search
            for the class. This is used if :attr:`class_or_name` is a string and
            the class cannot be located solely based on :attr:`class_or_name`.
            The first module in the list that contains the class
            is used.
        superclass (optional): A (list of) classes that the target class
            must inherit.

    Returns:
        The target class.

    Raises:
        ValueError: If class is not found based on :attr:`class_or_name` and
            :attr:`module_paths`.
        TypeError: If class does not inherits :attr:`superclass`.
    """
    class_ = class_or_name
    if is_str(class_):
        class_ = get_class(class_, module_path)
    if superclass is not None:
        if not issubclass(class_, superclass):
            raise TypeError(
                "A subclass of {} is expected. Got: {}".format(
                    superclass, class_))
    return class_

def get_class(class_name, module_paths=None):
    """Returns the class based on class name.

    Args:
        class_name (str): Name or full path to the class.
        module_paths (list): Paths to candidate modules to search for the
            class. This is used if the class cannot be located solely based on
            `class_name`. The first module in the list that contains the class
            is used.

    Returns:
        The target class.

    Raises:
        ValueError: If class is not found based on :attr:`class_name` and
            :attr:`module_paths`.
    """
    class_ = locate(class_name)
    if (class_ is None) and (module_paths is not None):
        for module_path in module_paths:
            #if module_path in _unimportable_modules:
            # Special treatment for unimportable modules by directly
            # accessing the class
            class_ = locate('.'.join([module_path, class_name]))
            if class_ is not None:
                break
            #else:
            #    module = importlib.import_module(module_path)
            #    if class_name in dir(module):
            #        class_ = getattr(module, class_name)
            #        break

    if class_ is None:
        raise ValueError(
            "Class not found in {}: {}".format(module_paths, class_name))

    return class_

def check_or_get_instance(ins_or_class_or_name, kwargs, module_paths=None,
                          classtype=None):
    """Returns a class instance and checks types.

    Args:
        ins_or_class_or_name: Can be of 3 types:

            - A string representing the name or full path to a class to \
              instantiate.
            - The class itself to instantiate.
            - The class instance itself to check types.

        kwargs (dict): Keyword arguments for the class constructor.
        module_paths (list, optional): Paths to candidate modules to
            search for the class. This is used if the class cannot be
            located solely based on :attr:`class_name`. The first module
            in the list that contains the class is used.
        classtype (optional): A (list of) classes of which the instance must
            be an instantiation.

    Raises:
        ValueError: If class is not found based on :attr:`class_name` and
            :attr:`module_paths`.
        ValueError: If :attr:`kwargs` contains arguments that are invalid
            for the class construction.
        TypeError: If the instance is not an instantiation of
            :attr:`classtype`.
    """
    ret = ins_or_class_or_name
    if is_str(ret) or isinstance(ret, type):
        ret = get_instance(ret, kwargs, module_paths)
    if classtype is not None:
        if not isinstance(ret, classtype):
            raise TypeError(
                "An instance of {} is expected. Got: {}".format(classtype, ret))
    return ret

def get_instance(class_or_name, kwargs, module_paths=None):
    """Creates a class instance.

    Args:
        class_or_name: Name or full path to a class to instantiate, or the
            class itself.
        kwargs (dict): Keyword arguments for the class constructor.
        module_paths (list, optional): Paths to candidate modules to
            search for the class. This is used if the class cannot be
            located solely based on :attr:`class_name`. The first module
            in the list that contains the class is used.

    Returns:
        A class instance.

    Raises:
        ValueError: If class is not found based on :attr:`class_or_name` and
            :attr:`module_paths`.
        ValueError: If :attr:`kwargs` contains arguments that are invalid
            for the class construction.
    """
    # Locate the class
    class_ = class_or_name
    if is_str(class_):
        class_ = get_class(class_, module_paths)
    # Check validity of arguments
    class_args = set(inspect.getargspec(class_.__init__).args)
    for key in kwargs.keys():
        if key not in class_args:
            raise ValueError(
                "Invalid argument for class %s.%s: %s, valid args:%s" %
                (class_.__module__, class_.__name__, key, class_args))

    return class_(**kwargs)


def get_instance_with_redundant_kwargs(
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
    class_args = set(inspect.getargspec(class_.__init__).args)
    for key, value in kwargs.items():
        if key in class_args:
            selected_kwargs[key] = value

    return class_(**selected_kwargs)


def get_function(fn_or_name, module_paths=None):
    """Returns the function of specified name and module.

    Args:
        fn_or_name (str or callable): Name or full path to a function, or the
            function itself.
        module_paths (list, optional): A list of paths to candidate modules to
            search for the function. This is used only when the function
            cannot be located solely based on :attr:`fn_or_name`. The first
            module in the list that contains the function is used.

    Returns:
        A function.
    """
    if is_callable(fn_or_name):
        return fn_or_name

    fn = locate(fn_or_name)
    if (fn is None) and (module_paths is not None):
        for module_path in module_paths:
            #if module_path in _unimportable_modules:
            fn = locate('.'.join([module_path, fn_or_name]))
            if fn is not None:
                break
            #module = importlib.import_module(module_path)
            #if fn_name in dir(module):
            #    fn = getattr(module, fn_name)
            #    break

    if fn is None:
        raise ValueError(
            "Method not found in {}: {}".format(module_paths, fn_or_name))

    return fn


def call_function_with_redundant_kwargs(fn, kwargs):
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

def get_args(fn):
    """Gets the arguments of a function.

    Args:
        fn (callable): The function to inspect.

    Returns:
        list: A list of argument names (str) of the function.
    """
    argspec = inspect.getargspec(fn)
    return argspec.args

def get_default_arg_values(fn):
    """Gets the arguments and respective default values of a function.

    Only arguments with default values are included in the output dictionary.

    Args:
        fn (callable): The function to inspect.

    Returns:
        dict: A dictionary that maps argument names (str) to their default
        values. The dictionary is empty if no arguments have default values.
    """
    argspec = inspect.getargspec(fn)
    if argspec.defaults is None:
        return {}
    num_defaults = len(argspec.defaults)
    return dict(zip(argspec.args[-num_defaults:], argspec.defaults))

def get_instance_kwargs(kwargs, hparams):
    """Makes a dict of keyword arguments with the following structure:

    `kwargs_ = {'hparams': dict(hparams), **kwargs}`.

    This is typically used for constructing a module which takes a set of
    arguments as well as a argument named `hparams`.

    Args:
        kwargs (dict): A dict of keyword arguments. Can be `None`.
        hparams: A dict or an instance of :class:`~texar.HParams` Can be `None`.

    Returns:
        A `dict` that contains the keyword arguments in :attr:`kwargs`, and
        an additional keyword argument named `hparams`.
    """
    if hparams is None or isinstance(hparams, dict):
        kwargs_ = {'hparams': hparams}
    elif isinstance(hparams, HParams):
        kwargs_ = {'hparams': hparams.todict()}
    else:
        raise ValueError(
            '`hparams` must be a dict, an instance of HParams, or a `None`.')
    kwargs_.update(kwargs or {})
    return kwargs_

def add_variable(variable, var_list):
    """Adds variable to a given list.

    Args:
        variable: A (list of) variable(s).
        var_list (list): The list where the :attr:`variable` are added.
    """
    if isinstance(variable, (list, tuple)):
        for var in variable:
            add_variable(var, var_list)
    else:
        if variable not in var_list:
            var_list.append(variable)

def get_unique_named_variable_scope(base_name):
    """Returns a variable scope with a unique name.

    Args:
        base_name (str): The base name to uniquified.

    Returns:
        An instance of :tf_main:`variable_scope <variable_scope>`.
    """
    with tf.variable_scope(None, default_name=base_name) as vs:
        return vs


def maybe_gloabl_mode(mode):
    """Returns :func:`texar.contex.global_mode` if :attr:`mode` is `None`,
    otherwise returns :attr:`mode` as-is.
    """
    if mode is None:
        return context.global_mode()
    else:
        return mode

def is_train_mode(mode):
    """Returns a bool Tensor indicating whether the global mode is TRAIN.
    If :attr:`mode` is `None`, the mode is determined by
    :func:`texar.contex.global_mode`.
    """
    if mode is None:
        return context.global_mode_train()
    else:
        return tf.equal(mode, tf.estimator.ModeKeys.TRAIN)

def is_eval_mode(mode):
    """Returns a bool Tensor indicating whether the global mode is EVAL.
    If :attr:`mode` is `None`, the mode is determined by
    :func:`texar.contex.global_mode`.
    """
    if mode is None:
        return context.global_mode_eval()
    else:
        return tf.equal(mode, tf.estimator.ModeKeys.EVAL)

def is_predict_mode(mode):
    """Returns a bool Tensor indicating whether the global mode is PREDICT.
    If :attr:`mode` is `None`, the mode is determined by
    :func:`texar.contex.global_mode`.
    """
    if mode is None:
        return context.global_mode_predict()
    else:
        return tf.equal(mode, tf.estimator.ModeKeys.PREDICT)

def is_train_mode_py(mode, default=True):
    """Returns a python boolean indicating whether the mode is TRAIN.

    Args:
        mode: A string taking value in
            :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`.
            Can be `None`.
        default (bool): The return value when :attr:`mode` is `None`. Default
            is `True`.

    Returns:
        A python boolean.
    """
    if mode is None:
        return default
    if mode not in context.valid_modes():
        raise ValueError('Unknown mode: {}'.format(mode))
    return mode == tf.estimator.ModeKeys.TRAIN

def is_eval_mode_py(mode, default=False):
    """Returns a python boolean indicating whether the mode is EVAL.

    Args:
        mode: A string taking value in
            :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`.
            Can be `None`.
        default (bool): The return value when :attr:`mode` is `None`. Default
            is `False`.

    Returns:
        A python boolean.
    """
    if mode is None:
        return default
    if mode not in context.valid_modes():
        raise ValueError('Unknown mode: {}'.format(mode))
    return mode == tf.estimator.ModeKeys.EVAL

def is_predict_mode_py(mode, default=False):
    """Returns a python boolean indicating whether the mode is PREDICT.

    Args:
        mode: A string taking value in
            :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`.
            Can be `None`.
        default (bool): The return value when :attr:`mode` is `None`. Default
            is `False`.

    Returns:
        A python boolean.
    """
    if mode is None:
        return default
    if mode not in context.valid_modes():
        raise ValueError('Unknown mode: {}'.format(mode))
    return mode == tf.estimator.ModeKeys.PREDICT

def switch_dropout(dropout_keep_prob, mode=None):
    """Turns off dropout when not in training mode.

    Args:
        dropout_keep_prob: Dropout keep probability in training mode
        mode (optional): A Tensor taking values of
            :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`.
            Dropout is activated if :attr:`mode` is `TRAIN`.
            If `None`, the mode is inferred from
            :func:`texar.context.global_mode`.

    Returns:
        A unit Tensor that equals the dropout keep probability in `TRAIN` mode,
        and `1.` in other modes.
    """
    return 1. - (1. - dropout_keep_prob) * tf.to_float(is_train_mode(mode))


def default_string(str_, default_str):
    """Returns :attr:`str_` if it is not `None` or empty, otherwise returns
    :attr:`default_str`.

    Args:
        str_: A string.
        default_str: A string.

    Returns:
        Either :attr:`str_` or :attr:`default_str`.
    """
    if str_ is not None and str_ != "":
        return str_
    else:
        return default_str

def patch_dict(tgt_dict, src_dict):
    """Recursively patch :attr:`tgt_dict` by adding items from :attr:`src_dict`
    that do not exist in :attr:`tgt_dict`.

    If respective items in :attr:`src_dict` and :attr:`tgt_dict` are both
    `dict`, the :attr:`tgt_dict` item is patched recursively.

    Args:
        tgt_dict (dict): Target dictionary to patch.
        src_dict (dict): Source dictionary.

    Return:
        dict: A patched dictionary.
    """
    patched_dict = copy.deepcopy(tgt_dict)
    for key, value in src_dict.items():
        if key not in patched_dict:
            patched_dict[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(patched_dict[key], dict):
            patched_dict[key] = patch_dict(patched_dict[key], value)
    return patched_dict

def fetch_subdict(src_dict, tgt_dict_or_keys):
    """Fetches a sub dict of :attr:`src_dict` with the keys in
    :attr:`tgt_dict_or_keys`.

    Args:
        src_dict: A dict or instance of :class:`texar.HParams`.
            The source dict to fetch values from.
        tgt_dict_or_keys: A dict, instance of :class:`texar.HParams`,
            or a list (or a dict_keys) of keys to be included in the output
            dict.

    Returns:
        A new dict that is a subdict of :attr:`src_dict`.
    """
    if src_dict is None:
        return src_dict

    if isinstance(tgt_dict_or_keys, HParams):
        tgt_dict_or_keys = tgt_dict_or_keys.todict()
    if isinstance(tgt_dict_or_keys, dict):
        tgt_dict_or_keys = tgt_dict_or_keys.keys()
    keys = list(tgt_dict_or_keys)

    if isinstance(src_dict, HParams):
        src_dict = src_dict.todict()

    return {k: src_dict[k] for k in keys if k in src_dict}

def uniquify_str(str_, str_set):
    """Uniquifies :attr:`str_` if :attr:`str_` is included in :attr:`str_set`.

    This is done by appending '_[digits]' to :attr:`str_`. Returns
    :attr:`str_` directly if :attr:`str_` is not included in :attr:`str_set`.

    Args:
        str_ (string): A string to uniquify.
        str_set (set, dict, or list): A collection of strings. The returned
            string is guaranteed to be different from the elements in the
            collection.

    Returns:
        string: The uniquified string. Returns :attr:`str_` directly if it is
            already unique.
    """
    if str_ not in str_set:
        return str_
    else:
        for i in range(1, len(str_set)+1):
            unique_str = str_ + "_%d" % i
            if unique_str not in str_set:
                return unique_str
    raise ValueError("Fails to uniquify string: " + str_)

def soft_sequence_embedding(embedding, soft_sequence):
    """Mixes sequences of soft vectors with a embedding tensor.

    Args:
        embedding: A Tensor of shape `[num_classes, emb_dim]` containing
            the embedding vectors.
        soft_sequence: A Tensor of shape `[batch_size, max_time, num_classes]`
            containing the weights (probabilities) of embedding vectors.

    Returns:
        A Tensor of shape `[batch_size, max_time, emb_dim]`

    Example::

        decoder_outputs, ... = decoder(...)
        soft_seq_emb = soft_sequence_embedding(
            tf.nn.softmax(decoder_outputs.logits), embedding)
    """
    return tf.tensordot(soft_sequence, embedding, [2, 0])

def straight_through(fw_tensor, bw_tensor):
    """Use a tensor in forward pass while backpropagating gradient to another.

    Args:
        fw_tensor: A tensor to be used in the forward pass.
        bw_tensor: A tensor to which gradient is backpropagated. Must have the
            same shape and type with :attr:`fw_tensor`.

    Returns:
        A tensor of the same shape and value with :attr:`fw_tensor` but will
            direct gradient to bw_tensor.
    """
    return tf.stop_gradient(fw_tensor) + bw_tensor - tf.stop_gradient(bw_tensor)

def ceildiv(a, b):
    """Divides with ceil.

    E.g., `5 / 2 = 2.5`, `ceildiv(5, 2) = 3`.

    Args:
        a (int): Dividend integer.
        b (int): Divisor integer.

    Returns:
        int: Ceil quotient.
    """
    return -(-a // b)

#TODO(haoran):is it appropriate to put shape_list function here?
def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)
    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)
    static = x.get_shape().as_list()
    shape = tf.shape(x)
    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret
