# Copyright 2019 The Texar Authors. All Rights Reserved.
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

import inspect
from pydoc import locate

import funcsigs

__all__ = [
    "get_args",
    "check_or_get_class",
    "get_class",
    "check_or_get_instance",
    "get_instance",
    "get_function",
    "uniquify_str",
]


def get_args(fn):
    r"""Gets the arguments of a function.

    Args:
        fn (callable): The function to inspect.

    Returns:
        list: A list of argument names (``str``) of the function.
    """
    argspec = inspect.getfullargspec(fn)
    args = argspec.args

    # Empty args can be because `fn` is decorated. Use `funcsigs.signature`
    # to re-do the inspect
    if len(args) == 0:
        args = funcsigs.signature(fn).parameters.keys()
        args = list(args)

    return args


def check_or_get_class(class_or_name, module_paths=None, superclass=None):
    r"""Returns the class and checks if the class inherits :attr:`superclass`.

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
    if isinstance(class_, str):
        class_ = get_class(class_, module_paths)
    if superclass is not None:
        if not issubclass(class_, superclass):
            raise TypeError(
                "A subclass of {} is expected. Got: {}".format(
                    superclass, class_))
    return class_


def get_class(class_name, module_paths=None):
    r"""Returns the class based on class name.

    Args:
        class_name (str): Name or full path to the class.
        module_paths (list): Paths to candidate modules to search for the
            class. This is used if the class cannot be located solely based on
            ``class_name``. The first module in the list that contains the class
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
            class_ = locate('.'.join([module_path, class_name]))
            if class_ is not None:
                break

    if class_ is None:
        raise ValueError(
            "Class not found in {}: {}".format(module_paths, class_name))

    return class_


def check_or_get_instance(ins_or_class_or_name, kwargs, module_paths=None,
                          classtype=None):
    r"""Returns a class instance and checks types.

    Args:
        ins_or_class_or_name: Can be of 3 types:

            - A class to instantiate.
            - A string of the name or full path to a class to instantiate.
            - The class instance to check types.

        kwargs (dict): Keyword arguments for the class constructor. Ignored
            if ``ins_or_class_or_name`` is a class instance.
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
    if isinstance(ret, (str, type)):
        ret = get_instance(ret, kwargs, module_paths)
    if classtype is not None:
        if not isinstance(ret, classtype):
            raise TypeError(
                "An instance of {} is expected. Got: {}".format(classtype, ret))
    return ret


def get_instance(class_or_name, kwargs, module_paths=None):
    r"""Creates a class instance.

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
    if isinstance(class_, str):
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


def get_function(fn_or_name, module_paths=None):
    r"""Returns the function of specified name and module.

    Args:
        fn_or_name (str or callable): Name or full path to a function, or the
            function itself.
        module_paths (list, optional): A list of paths to candidate modules to
            search for the function. This is used only when the function
            cannot be located solely based on :attr:`fn_or_name`. The first
            module in the list that contains the function is used.

    Returns:
        A function.

    Raises:
        ValueError: If method with name as :attr:`fn_or_name` is not found.
    """
    if callable(fn_or_name):
        return fn_or_name

    fn = locate(fn_or_name)
    if (fn is None) and (module_paths is not None):
        for module_path in module_paths:
            fn = locate('.'.join([module_path, fn_or_name]))
            if fn is not None:
                break

    if fn is None:
        raise ValueError(
            "Method not found in {}: {}".format(module_paths, fn_or_name))

    return fn


def uniquify_str(str_, str_set):
    r"""Uniquifies :attr:`str_` if :attr:`str_` is included in :attr:`str_set`.

    This is done by appending a number to :attr:`str_`. Returns
    :attr:`str_` directly if it is not included in :attr:`str_set`.

    Args:
        str\_ (string): A string to uniquify.
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
        for i in range(1, len(str_set) + 1):
            unique_str = str_ + "_%d" % i
            if unique_str not in str_set:
                return unique_str
    raise ValueError("Failed to uniquify string: " + str_)
