#
"""
Miscellaneous Utility functions.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# pylint: disable=invalid-name, no-member, no-name-in-module, protected-access
# pylint: disable=redefined-outer-name, too-many-arguments

import inspect
from pydoc import locate
import copy
import collections
import numpy as np

import tensorflow as tf

from texar.hyperparams import HParams
from texar.utils.dtypes import is_str, is_callable, compat_as_text, \
        _maybe_list_to_array

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
    "check_or_get_instance_with_redundant_kwargs",
    "get_instance_with_redundant_kwargs",
    "get_function",
    "call_function_with_redundant_kwargs",
    "get_args",
    "get_default_arg_values",
    "get_instance_kwargs",
    "dict_patch",
    "dict_lookup",
    "dict_fetch",
    "dict_pop",
    "flatten_dict",
    "strip_token",
    "strip_eos",
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

def check_or_get_instance_with_redundant_kwargs(
        ins_or_class_or_name, kwargs, module_paths=None, classtype=None):
    """Returns a class instance and checks types.

    Only those keyword arguments in :attr:`kwargs` that are included in the
    class construction method are used.

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
        fn (function): A callable. If :attr:`fn` is not a python function,
            :attr:`fn.__call__` is called.
        kwargs (dict): A `dict` of arguments for the callable. It
            may include invalid arguments which will be ignored.

    Returns:
        The returned results by calling :attr:`fn`.
    """
    try:
        fn_args = set(inspect.getargspec(fn).args)
    except TypeError:
        fn_args = set(inspect.getargspec(fn.__call__).args)

    # Select valid arguments
    selected_kwargs = {}
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
    """Looks up :attr:`keys` in the dict, outputs the corresponding values.

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
            E.g., if :attr:`sep``='.'`, then { "a": { "b": 3 } } is converted
            into { "a.b": 3 }.

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

def strip_token(str_, token, compat=True):
    """Returns a copy of the strings with leading and trailing tokens
    removed.

    Assumes tokens in the strings are separated with the space character.

    Args:
        str_: A `str`, or an `n`-D numpy array or (possibly nested)
            list of `str`.
        token (str): The token to strip, e.g., the '<PAD>' token defined in
            :class:`~texar.data.vocabulary.SpecialTokens`.PAD
        compat (bool): Whether to convert tokens into `unicode` (Python 2)
            or `str` (Python 3).

    Returns:
        The stripped strings of the same structure/shape as :attr:`str_`.
    """
    def _recur_strip(s):
        if is_str(s):
            return ' '.join(s.strip().split()).\
                replace(' '+token, '').replace(token+' ', '')
        else:
            s_ = [_recur_strip(si) for si in s]
            return _maybe_list_to_array(s_, s)

    if compat:
        str_ = compat_as_text(str_)

    strp_str = _recur_strip(str_)

    #if isinstance(str_, (list, tuple)):
    #    return type(str_)(strp_str)
    #else:
    #    return np.asarray(strp_str)
    return strp_str

def strip_eos(str_, eos_token='<EOS>', compat=True):
    """Remove the EOS token and all subsequent tokens.

    Assumes tokens in the strings are separated with the space character.

    Args:
        str_: A `str`, or an `n`-D numpy array or (possibly nested)
            list of `str`.
        eos_token (str): The EOS token. Default is '<EOS>' as defined in
            :class:`~texar.data.vocabulary.SpecialTokens`.`EOS`
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

    if compat:
        str_ = compat_as_text(str_)

    strp_str = _recur_strip(str_)

    #if isinstance(str_, (list, tuple)):
    #    return type(str_)(strp_str)
    #else:
    #    return np.asarray(strp_str)
    return strp_str
_strip_eos_ = strip_eos

def strip_bos(str_, bos_token='<BOS>', compat=True):
    """Remove the leading BOS token.

    Assumes tokens in the strings are separated with the space character.

    Args:
        str_: A `str`, or an `n`-D numpy array or (possibly nested)
            list of `str`.
        bos_token (str): The BOS token. Default is '<BOS>' as defined in
            :class:`~texar.data.vocabulary.SpecialTokens`.`BOS`
        compat (bool): Whether to convert tokens into `unicode` (Python 2)
            or `str` (Python 3).

    Returns:
        Strings of the same structure/shape as :attr:`str_`.
    """
    def _recur_strip(s):
        if is_str(s):
            return ' '.join(s.strip().split()).replace(bos_token+' ', '')
        else:
            s_ = [_recur_strip(si) for si in s]
            return _maybe_list_to_array(s_, s)

    if compat:
        str_ = compat_as_text(str_)

    strp_str = _recur_strip(str_)

    #if isinstance(str_, (list, tuple)):
    #    return type(str_)(strp_str)
    #else:
    #    return np.asarray(strp_str)
    return strp_str
_strip_bos_ = strip_bos

def strip_special_tokens(str_, strip_pad='<PAD>', strip_bos='<BOS>',
                         strip_eos='<EOS>', compat=True):
    """Removes special tokens of strings, including:

        - Removes EOS and all subsequent tokens
        - Removes leading and and trailing PAD tokens
        - Removes leading BOS tokens

    Args:
        str_: A `str`, or an `n`-D numpy array or (possibly nested)
            list of `str`.
        strip_pad (str): The PAD token to strip from the strings (i.e., remove
            the leading and trailing PAD tokens of the strings). Default
            is '<PAD>' as defined in
            :class:`~texar.data.vocabulary.SpecialTokens`.`PAD`.
            Set to `None` to disable the stripping.
        strip_bos (str): The BOS token to strip from the strings (i.e., remove
            the leading BOS tokens of the strings).
            Default is '<BOS>' as defined in
            :class:`~texar.data.vocabulary.SpecialTokens`.`BOS`.
            Set to `None` to disable the stripping.
        strip_eos (str): The EOS token to strip from the strings (i.e., remove
            the EOS tokens and all subsequent tokens of the strings).
            Default is '<EOS>' as defined in
            :class:`~texar.data.vocabulary.SpecialTokens`.`EOS`.
            Set to `None` to disable the stripping.
        compat (bool): Whether to convert tokens into `unicode` (Python 2)
            or `str` (Python 3).

    Returns:
        Strings of the same shape of :attr:`str_` with special tokens stripped.
    """
    if compat:
        str_ = compat_as_text(str_)

    if strip_eos is not None:
        str_ = _strip_eos_(str_, strip_eos, compat=False)

    if strip_pad is not None:
        str_ = strip_token(str_, strip_pad, compat=False)

    if strip_bos is not None:
        str_ = _strip_bos_(str_, strip_bos, compat=False)

    return str_

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

    #if isinstance(tokens, (list, tuple)):
    #    return type(tokens)(str_)
    #else:
    #    return np.asarray(str_)
    return str_

def map_ids_to_strs(ids, vocab, join=True, strip_pad='<PAD>',
                    strip_bos='<BOS>', strip_eos='<EOS>', compat=True):
    """Transforms indexes to strings by id-token mapping, token concat, token
    stripping, etc.

    Args:
        ids: An n-D numpy array or (possibly nested) list of `int` indexes.
        vocab: An instance of :class:`~texar.data.Vocab`.
        join (bool): Whether concat along the last dimension of :attr:`ids`
            the tokens into a string with a space character.
        strip_pad (str): The PAD token to strip from the strings (i.e., remove
            the leading and trailing PAD tokens of the strings). Default
            is '<PAD>' as defined in
            :class:`~texar.data.vocabulary.SpecialTokens`.`PAD`.
            Set to `None` to disable the stripping.
        strip_bos (str): The BOS token to strip from the strings (i.e., remove
            the leading BOS tokens of the strings).
            Default is '<BOS>' as defined in
            :class:`~texar.data.vocabulary.SpecialTokens`.`BOS`.
            Set to `None` to disable the stripping.
        strip_eos (str): The EOS token to strip from the strings (i.e., remove
            the EOS tokens and all subsequent tokens of the strings).
            Default is '<EOS>' as defined in
            :class:`~texar.data.vocabulary.SpecialTokens`.`EOS`.
            Set to `None` to disable the stripping.
    Returns:
        If :attr:`join`=True, returns a (n-1)-D numpy array (or list) of
        concatenated strings. If :attr:`join`=False, returns an n-D numpy
        array (or list) of str tokens.
    """
    tokens = vocab.map_ids_to_tokens_py(ids)

    if compat:
        tokens = compat_as_text(tokens)

    str_ = str_join(tokens, compat=False)

    str_ = strip_special_tokens(
        str_, strip_pad=strip_pad, strip_bos=strip_bos, strip_eos=strip_eos,
        compat=False)

    def _recur_split(s):
        if is_str(s):
            return _maybe_list_to_array(s.split(), str_)
        else:
            s_ = [_recur_split(si) for si in s]
            return _maybe_list_to_array(s_, s)

    if join:
        return str_
    else:
        return _recur_split(str_)

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
