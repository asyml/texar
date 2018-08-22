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
Utility functions related to variables.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# pylint: disable=invalid-name

import tensorflow as tf

__all__ = [
    "get_unique_named_variable_scope",
    "add_variable",
    "collect_trainable_variables"
]


def get_unique_named_variable_scope(base_name):
    """Returns a variable scope with a unique name.

    Args:
        base_name (str): The base name to uniquified.

    Returns:
        An instance of :tf_main:`variable_scope <variable_scope>`.

    Example:

        .. code-block:: python

            vs = get_unique_named_variable_scope('base_name')
            with tf.variable_scope(vs):
                ....
    """
    with tf.variable_scope(None, default_name=base_name) as vs:
        return vs

def add_variable(variable, var_list):
    """Adds variable to a given list.

    Args:
        variable: A (list of) variable(s).
        var_list (list): The list where the :attr:`variable` are added to.
    """
    if isinstance(variable, (list, tuple)):
        for var in variable:
            add_variable(var, var_list)
    else:
        if variable not in var_list:
            var_list.append(variable)

def collect_trainable_variables(modules):
    """Collects all trainable variables of modules.

    Trainable variables included in multiple modules occur only once in the
    returned list.

    Args:
        modules: A (list of) instance of the subclasses of
            :class:`~texar.modules.ModuleBase`.

    Returns:
        A list of trainable variables in the modules.
    """
    if not isinstance(modules, (list, tuple)):
        modules = [modules]

    var_list = []
    for mod in modules:
        add_variable(mod.trainable_variables, var_list)

    return var_list
