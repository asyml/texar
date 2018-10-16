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
Miscellaneous Utility functions.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# pylint: disable=invalid-name, no-member, no-name-in-module, protected-access
# pylint: disable=redefined-outer-name, too-many-arguments

import inspect
import funcsigs
from pydoc import locate
import copy
import collections
import numpy as np

import tensorflow as tf

from texar.hyperparams import HParams
from texar.utils.dtypes import is_str, is_callable, compat_as_text, \
        _maybe_list_to_array

# pylint: disable=anomalous-backslash-in-string

MAX_SEQ_LENGTH = np.iinfo(np.int32).max

## Some modules cannot be imported directly,
## e.g., `import tensorflow.train` fails.
## Such modules are treated in a special way in utils like `get_class` as below.
#_unimportable_modules = {
#    'tensorflow.train', 'tensorflow.keras.regularizers'
#}

__all__ = [
    "_inspect_getargspec",
    "get_args",
    "get_default_arg_values",
    "check_or_get_class",
    "get_class",
    "check_or_get_instance",
    "get_instance",
    "check_or_get_instance_with_redundant_kwargs",
    "get_instance_with_redundant_kwargs",
    "get_function",
    "call_function_with_redundant_kwargs",
    "get_instance_kwargs",
    "dict_patch",
    "dict_lookup",
    "dict_fetch",
    "dict_pop",
    "flatten_dict",
    "strip_token",
    "strip_eos",
    "strip_bos",
    "strip_special_tokens",
    "str_join",
    "map_ids_to_strs",
    "default_str",
    "uniquify_str",
    "ceildiv",
    "straight_through"
]


# TODO(zhiting): complete this
def _expand_name(name):
    """Replaces common shorthands with respective full names.

        "tf.xxx" --> "tensorflow.xxx"
        "tx.xxx" --> "texar.xxx"
    """
    return name

def _inspect_getargspec(fn):
    """Returns `inspect.getargspec(fn)` for Py2 and `inspect.getfullargspec(fn)`
    for Py3
    """
    try:
        return inspect.getfullargspec(fn)
    except AttributeError:
        return inspect.getargspec(fn)

def get_args(fn):
    """Gets the arguments of a function.

    Args:
        fn (callable): The function to inspect.

    Returns:
        list: A list of argument names (str) of the function.
    """
    argspec = _inspect_getargspec(fn)
    args = argspec.args

    # Empty args can be because `fn` is decorated. Use `funcsigs.signature`
    # to re-do the inspect
    if len(args) == 0:
        args = funcsigs.signature(fn).parameters.keys()
        args = list(args)

    return args

def get_default_arg_values(fn):
    """Gets the arguments and respective default values of a function.

    Only arguments with default values are included in the output dictionary.

    Args:
        fn (callable): The function to inspect.

    Returns:
        dict: A dictionary that maps argument names (str) to their default
        values. The dictionary is empty if no arguments have default values.
    """
    argspec = _inspect_getargspec(fn)
    if argspec.defaults is None:
        return {}
    num_defaults = len(argspec.defaults)
    return dict(zip(argspec.args[-num_defaults:], argspec.defaults))


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

            - A class to instantiate.
            - A string of the name or full path to a class to \
              instantiate.
            - The class instance to check types.

        kwargs (dict): Keyword arguments for the class constructor. Ignored
            if `ins_or_class_or_name` is a class instance.
        module_paths (list, optional): Paths to candidate modules to
            search for the class. This is used if the class cannot be
            located solely based on :attr:`class_name`. The first module
            in the list that contains the class is used.
        classtype (optional): A (list of) class of which the instance must
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
        class_or_name: A class, or its name or full path to a class to
            instantiate.
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
    class_args = set(get_args(class_.__init__))

    if kwargs is None:
        kwargs = {}
    for key in kwargs.keys():
        if key not in class_args:
            raise ValueError(
                "Invalid argument for class %s.%s: %s, valid args: %s" %
                (class_.__module__, class_.__name__, key, list(class_args)))

    return class_(**kwargs)

def check_or_get_instance_with_redundant_kwargs(
        ins_or_class_or_name, kwargs, module_paths=None, classtype=None):
    """Returns a class instance and checks types.

    Only those keyword arguments in :attr:`kwargs` that are included in the
    class construction method are used.

    Args:
        ins_or_class_or_name: Can be of 3 types:

            - A class to instantiate.
            - A string of the name or module path to a class to \
              instantiate.
            - The class instance to check types.

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
        ret = get_instance_with_redundant_kwargs(ret, kwargs, module_paths)
    if classtype is not None:
        if not isinstance(ret, classtype):
            raise TypeError(
                "An instance of {} is expected. Got: {}".format(classtype, ret))
    return ret

def get_instance_with_redundant_kwargs(
        class_name, kwargs, module_paths=None):
    """Creates a class instance.

    Only those keyword arguments in :attr:`kwargs` that are included in the
    class construction method are used.

    Args:
        class_name (str): A class or its name or module path.
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
    class_args = set(get_args(class_.__init__))
    if kwargs is None:
        kwargs = {}
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
        fn (function): A callable. If :attr:`fn` is not a python function,
            :attr:`fn.__call__` is called.
        kwargs (dict): A `dict` of arguments for the callable. It
            may include invalid arguments which will be ignored.

    Returns:
        The returned results by calling :attr:`fn`.
    """
    try:
        fn_args = set(get_args(fn))
    except TypeError:
        fn_args = set(get_args(fn.__cal__))

    if kwargs is None:
        kwargs = {}

    # Select valid arguments
    selected_kwargs = {}
    for key, value in kwargs.items():
        if key in fn_args:
            selected_kwargs[key] = value

    return fn(**selected_kwargs)


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

def dict_patch(tgt_dict, src_dict):
    """Recursively patch :attr:`tgt_dict` by adding items from :attr:`src_dict`
    that do not exist in :attr:`tgt_dict`.

    If respective items in :attr:`src_dict` and :attr:`tgt_dict` are both
    `dict`, the :attr:`tgt_dict` item is patched recursively.

    Args:
        tgt_dict (dict): Target dictionary to patch.
        src_dict (dict): Source dictionary.

    Return:
        dict: The new :attr:`tgt_dict` that is patched.
    """
    if src_dict is None:
        return tgt_dict

    for key, value in src_dict.items():
        if key not in tgt_dict:
            tgt_dict[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(tgt_dict[key], dict):
            tgt_dict[key] = dict_patch(tgt_dict[key], value)
    return tgt_dict

def dict_lookup(dict_, keys, default=None):
    """Looks up :attr:`keys` in the dict, returns the corresponding values.

    The :attr:`default` is used for keys not present in the dict.

    Args:
        dict_ (dict): A dictionary for lookup.
        keys: A numpy array or a (possibly nested) list of keys.
        default (optional): Value to be returned when a key is not in
            :attr:`dict_`. Error is raised if :attr:`default` is not given and
            key is not in the dict.

    Returns:
        A numpy array of values with the same structure as :attr:`keys`.

    Raises:
        TypeError: If key is not in :attr:`dict_` and :attr:`default` is `None`.
    """
    return np.vectorize(lambda x: dict_.get(x, default))(keys)

def dict_fetch(src_dict, tgt_dict_or_keys):
    """Fetches a sub dict of :attr:`src_dict` with the keys in
    :attr:`tgt_dict_or_keys`.

    Args:
        src_dict: A dict or instance of :class:`~texar.HParams`.
            The source dict to fetch values from.
        tgt_dict_or_keys: A dict, instance of :class:`~texar.HParams`,
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

def dict_pop(dict_, pop_keys, default=None):
    """Removes keys from a dict and returns their values.

    Args:
        dict_ (dict): A dictionary from which items are removed.
        pop_keys: A key or a list of keys to remove and return respective
            values or :attr:`default`.
        default (optional): Value to be returned when a key is not in
            :attr:`dict_`. The default value is `None`.

    Returns:
        A `dict` of the items removed from :attr:`dict_`.
    """
    if not isinstance(pop_keys, (list, tuple)):
        pop_keys = [pop_keys]
    ret_dict = {key: dict_.pop(key, default) for key in pop_keys}
    return ret_dict

def flatten_dict(dict_, parent_key="", sep="."):
    """Flattens a nested dictionary. Namedtuples within the dictionary are
    converted to dicts.

    Adapted from:
    https://github.com/google/seq2seq/blob/master/seq2seq/models/model_base.py

    Args:
        dict_ (dict): The dictionary to flatten.
        parent_key (str): A prefix to prepend to each key.
        sep (str): Separator that intervenes between parent and child keys.
            E.g., if `sep` == '.', then `{ "a": { "b": 3 } }` is converted
            into `{ "a.b": 3 }`.

    Returns:
        A new flattened `dict`.
    """
    items = []
    for key, value in dict_.items():
        key_ = parent_key + sep + key if parent_key else key
        if isinstance(value, collections.MutableMapping):
            items.extend(flatten_dict(value, key_, sep=sep).items())
        elif isinstance(value, tuple) and hasattr(value, "_asdict"):
            dict_items = collections.OrderedDict(zip(value._fields, value))
            items.extend(flatten_dict(dict_items, key_, sep=sep).items())
        else:
            items.append((key_, value))
    return dict(items)

def default_str(str_, default_str):
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

def uniquify_str(str_, str_set):
    """Uniquifies :attr:`str_` if :attr:`str_` is included in :attr:`str_set`.

    This is done by appending a number to :attr:`str_`. Returns
    :attr:`str_` directly if it is not included in :attr:`str_set`.

    Args:
        str_ (string): A string to uniquify.
        str_set (set, dict, or list): A collection of strings. The returned
            string is guaranteed to be different from the elements in the
            collection.

    Returns:
        The uniquified string. Returns :attr:`str_` directly if it is
        already unique.

    Example:

        .. code-block:: python

            print(uniquify_str('name', ['name', 'name_1']))
            # 'name_2'

    """
    if str_ not in str_set:
        return str_
    else:
        for i in range(1, len(str_set)+1):
            unique_str = str_ + "_%d" % i
            if unique_str not in str_set:
                return unique_str
    raise ValueError("Fails to uniquify string: " + str_)


def _recur_split(s, dtype_as):
    """Splits (possibly nested list of) strings recursively.
    """
    if is_str(s):
        return _maybe_list_to_array(s.split(), dtype_as)
    else:
        s_ = [_recur_split(si, dtype_as) for si in s]
        return _maybe_list_to_array(s_, s)


def strip_token(str_, token, is_token_list=False, compat=True):
    """Returns a copy of strings with leading and trailing tokens removed.

    Note that besides :attr:`token`, all leading and trailing whitespace
    characters are also removed.

    If :attr:`is_token_list` is False, then the function assumes tokens in
    :attr:`str_` are separated with whitespace character.

    Args:
        str\_: A `str`, or an `n`-D numpy array or (possibly nested)
            list of `str`.
        token (str): The token to strip, e.g., the '<PAD>' token defined in
            :class:`~texar.data.SpecialTokens`.PAD
        is_token_list (bool): Whether each sentence in :attr:`str_` is a list
            of tokens. If False, each sentence in :attr:`str_` is assumed to
            contain tokens separated with space character.
        compat (bool): Whether to convert tokens into `unicode` (Python 2)
            or `str` (Python 3).

    Returns:
        The stripped strings of the same structure/shape as :attr:`str_`.

    Example:

        .. code-block:: python

            str_ = '<PAD> a sentence <PAD> <PAD>  '
            str_stripped = strip_token(str_, '<PAD>')
            # str_stripped == 'a sentence'

            str_ = ['<PAD>', 'a', 'sentence', '<PAD>', '<PAD>', '', '']
            str_stripped = strip_token(str_, '<PAD>', is_token_list=True)
            # str_stripped == 'a sentence'
    """
    def _recur_strip(s):
        if is_str(s):
            if token == "":
                return ' '.join(s.strip().split())
            else:
                return ' '.join(s.strip().split()).\
                    replace(' '+token, '').replace(token+' ', '')
        else:
            s_ = [_recur_strip(si) for si in s]
            return _maybe_list_to_array(s_, s)

    s = str_

    if compat:
        s = compat_as_text(s)

    if is_token_list:
        s = str_join(s, compat=False)

    strp_str = _recur_strip(s)

    if is_token_list:
        strp_str = _recur_split(strp_str, str_)

    return strp_str

def strip_eos(str_, eos_token='<EOS>', is_token_list=False, compat=True):
    """Remove the EOS token and all subsequent tokens.

    If :attr:`is_token_list` is False, then the function assumes tokens in
    :attr:`str_` are separated with whitespace character.

    Args:
        str\_: A `str`, or an `n`-D numpy array or (possibly nested)
            list of `str`.
        eos_token (str): The EOS token. Default is '<EOS>' as defined in
            :class:`~texar.data.SpecialTokens`.EOS
        is_token_list (bool): Whether each sentence in :attr:`str_` is a list
            of tokens. If False, each sentence in :attr:`str_` is assumed to
            contain tokens separated with space character.
        compat (bool): Whether to convert tokens into `unicode` (Python 2)
            or `str` (Python 3).

    Returns:
        Strings of the same structure/shape as :attr:`str_`.
    """
    def _recur_strip(s):
        if is_str(s):
            s_tokens = s.split()
            if eos_token in s_tokens:
                return ' '.join(s_tokens[:s_tokens.index(eos_token)])
            else:
                return s
        else:
            s_ = [_recur_strip(si) for si in s]
            return _maybe_list_to_array(s_, s)

    s = str_

    if compat:
        s = compat_as_text(s)

    if is_token_list:
        s = str_join(s, compat=False)

    strp_str = _recur_strip(s)

    if is_token_list:
        strp_str = _recur_split(strp_str, str_)

    return strp_str
_strip_eos_ = strip_eos

def strip_bos(str_, bos_token='<BOS>', is_token_list=False, compat=True):
    """Remove all leading BOS tokens.

    Note that besides :attr:`bos_token`, all leading and trailing whitespace
    characters are also removed.

    If :attr:`is_token_list` is False, then the function assumes tokens in
    :attr:`str_` are separated with whitespace character.

    Args:
        str\_: A `str`, or an `n`-D numpy array or (possibly nested)
            list of `str`.
        bos_token (str): The BOS token. Default is '<BOS>' as defined in
            :class:`~texar.data.SpecialTokens`.BOS
        is_token_list (bool): Whether each sentence in :attr:`str_` is a list
            of tokens. If False, each sentence in :attr:`str_` is assumed to
            contain tokens separated with space character.
        compat (bool): Whether to convert tokens into `unicode` (Python 2)
            or `str` (Python 3).

    Returns:
        Strings of the same structure/shape as :attr:`str_`.
    """
    def _recur_strip(s):
        if is_str(s):
            if bos_token == '':
                return ' '.join(s.strip().split())
            else:
                return ' '.join(s.strip().split()).replace(bos_token+' ', '')
        else:
            s_ = [_recur_strip(si) for si in s]
            return _maybe_list_to_array(s_, s)

    s = str_

    if compat:
        s = compat_as_text(s)

    if is_token_list:
        s = str_join(s, compat=False)

    strp_str = _recur_strip(s)

    if is_token_list:
        strp_str = _recur_split(strp_str, str_)

    return strp_str
_strip_bos_ = strip_bos

def strip_special_tokens(str_, strip_pad='<PAD>', strip_bos='<BOS>',
                         strip_eos='<EOS>', is_token_list=False, compat=True):
    """Removes special tokens in strings, including:

        - Removes EOS and all subsequent tokens
        - Removes leading and and trailing PAD tokens
        - Removes leading BOS tokens

    Note that besides the special tokens, all leading and trailing whitespace
    characters are also removed.

    This is a joint function of :func:`strip_eos`, :func:`strip_pad`, and
    :func:`strip_bos`

    Args:
        str\_: A `str`, or an `n`-D numpy array or (possibly nested)
            list of `str`.
        strip_pad (str): The PAD token to strip from the strings (i.e., remove
            the leading and trailing PAD tokens of the strings). Default
            is '<PAD>' as defined in
            :class:`~texar.data.SpecialTokens`.PAD.
            Set to `None` or `False` to disable the stripping.
        strip_bos (str): The BOS token to strip from the strings (i.e., remove
            the leading BOS tokens of the strings).
            Default is '<BOS>' as defined in
            :class:`~texar.data.SpecialTokens`.BOS.
            Set to `None` or `False` to disable the stripping.
        strip_eos (str): The EOS token to strip from the strings (i.e., remove
            the EOS tokens and all subsequent tokens of the strings).
            Default is '<EOS>' as defined in
            :class:`~texar.data.SpecialTokens`.EOS.
            Set to `None` or `False` to disable the stripping.
        is_token_list (bool): Whether each sentence in :attr:`str_` is a list
            of tokens. If False, each sentence in :attr:`str_` is assumed to
            contain tokens separated with space character.
        compat (bool): Whether to convert tokens into `unicode` (Python 2)
            or `str` (Python 3).

    Returns:
        Strings of the same shape of :attr:`str_` with special tokens stripped.
    """
    s = str_

    if compat:
        s = compat_as_text(s)

    if is_token_list:
        s = str_join(s, compat=False)

    if strip_eos is not None and strip_eos is not False:
        s = _strip_eos_(s, strip_eos, is_token_list=False, compat=False)

    if strip_pad is not None and strip_pad is not False:
        s = strip_token(s, strip_pad, is_token_list=False, compat=False)

    if strip_bos is not None and strip_bos is not False:
        s = _strip_bos_(s, strip_bos, is_token_list=False, compat=False)

    if is_token_list:
        s = _recur_split(s, str_)

    return s

def str_join(tokens, sep=' ', compat=True):
    """Concats :attr:`tokens` along the last dimension with intervening
    occurrences of :attr:`sep`.

    Args:
        tokens: An `n`-D numpy array or (possibly nested) list of `str`.
        sep (str): The string intervening between the tokens.
        compat (bool): Whether to convert tokens into `unicode` (Python 2)
            or `str` (Python 3).

    Returns:
        An `(n-1)`-D numpy array (or list) of `str`.
    """
    def _recur_join(s):
        if len(s) == 0:
            return ''
        elif is_str(s[0]):
            return sep.join(s)
        else:
            s_ = [_recur_join(si) for si in s]
            return _maybe_list_to_array(s_, s)

    if compat:
        tokens = compat_as_text(tokens)

    str_ = _recur_join(tokens)

    return str_

def map_ids_to_strs(ids, vocab, join=True, strip_pad='<PAD>',
                    strip_bos='<BOS>', strip_eos='<EOS>', compat=True):
    """Transforms `int` indexes to strings by mapping ids to tokens,
    concatenating tokens into sentences, and stripping special tokens, etc.

    Args:
        ids: An n-D numpy array or (possibly nested) list of `int` indexes.
        vocab: An instance of :class:`~texar.data.Vocab`.
        join (bool): Whether to concat along the last dimension of the
            the tokens into a string separated with a space character.
        strip_pad (str): The PAD token to strip from the strings (i.e., remove
            the leading and trailing PAD tokens of the strings). Default
            is '<PAD>' as defined in
            :class:`~texar.data.SpecialTokens`.PAD.
            Set to `None` or `False` to disable the stripping.
        strip_bos (str): The BOS token to strip from the strings (i.e., remove
            the leading BOS tokens of the strings).
            Default is '<BOS>' as defined in
            :class:`~texar.data.SpecialTokens`.BOS.
            Set to `None` or `False` to disable the stripping.
        strip_eos (str): The EOS token to strip from the strings (i.e., remove
            the EOS tokens and all subsequent tokens of the strings).
            Default is '<EOS>' as defined in
            :class:`~texar.data.SpecialTokens`.EOS.
            Set to `None` or `False` to disable the stripping.

    Returns:
        If :attr:`join` is True, returns a `(n-1)`-D numpy array (or list) of
        concatenated strings. If :attr:`join` is False, returns an `n`-D numpy
        array (or list) of str tokens.

    Example:

        .. code-block:: python

            text_ids = [[1, 9, 6, 2, 0, 0], [1, 28, 7, 8, 2, 0]]

            text = map_ids_to_strs(text_ids, data.vocab)
            # text == ['a sentence', 'parsed from ids']

            text = map_ids_to_strs(
                text_ids, data.vocab, join=False,
                strip_pad=None, strip_bos=None, strip_eos=None)
            # text == [['<BOS>', 'a', 'sentence', '<EOS>', '<PAD>', '<PAD>'],
            #          ['<BOS>', 'parsed', 'from', 'ids', '<EOS>', '<PAD>']]
    """
    tokens = vocab.map_ids_to_tokens_py(ids)
    if isinstance(ids, (list, tuple)):
        tokens = tokens.tolist()

    if compat:
        tokens = compat_as_text(tokens)

    str_ = str_join(tokens, compat=False)

    str_ = strip_special_tokens(
        str_, strip_pad=strip_pad, strip_bos=strip_bos, strip_eos=strip_eos,
        compat=False)

    if join:
        return str_
    else:
        return _recur_split(str_, ids)

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
