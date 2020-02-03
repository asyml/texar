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
Base class for modules.
"""

import re

import tensorflow as tf

from texar.tf.utils.exceptions import TexarError
from texar.tf.hyperparams import HParams

__all__ = [
    "ModuleBase"
]


class ModuleBase(object):
    """Base class inherited by modules that create Variables and are
    configurable through hyperparameters.

    A Texar module inheriting :class:`~texar.tf.ModuleBase` has following key
    features:

        - **Convenient variable re-use**: A module instance creates \
        its own sets of variables, and automatically re-uses its variables on \
        subsequent calls. Hence TF variable/name scope is \
        transparent to users. For example:

            .. code-block:: python

                encoder = UnidirectionalRNNEncoder(hparams) # create instance
                output_1 = encoder(inputs_1) # variables are created
                output_2 = encoder(inputs_2) # variables are re-used

                print(encoder.trainable_variables) # access trainable variables
                # [ ... ]

        - **Configurable through hyperparameters**: Each module defines \
        allowed hyperparameters and default values. Hyperparameters not \
        specified by users will take default values.

        - **Callable**: As the above example, a module instance is "called" \
        with input tensors and returns output tensors. Every call of a module \
        will add ops to the Graph to perform the module's logic.

    Args:
        hparams (dict, optional): Hyperparameters of the module. See
            :meth:`default_hparams` for the structure and default values.


    .. document private functions
    .. automethod:: _build
    """

    def __init__(self, hparams=None):
        if not hasattr(self, '_hparams'):
            self._hparams = HParams(hparams, self.default_hparams())
        else:
            # Probably already parsed by subclasses. We rely on subclass
            # implementations to get this right.
            # As a sanity check, we require `hparams` to be `None` in this case.
            if hparams is not None:
                raise ValueError(
                    "`self._hparams` already exists. Argument `hparams` "
                    "must be set to `None` in this case.")
        self._template = tf.make_template(self._hparams.name, self._build,
                                          create_scope_now_=True)
        self._unique_name = self.variable_scope.name.split("/")[-1]
        self._trainable_variables = []
        self._built = False

    @staticmethod
    def default_hparams():
        """Returns a `dict` of hyperparameters of the module with default
        values. Used to replace the missing values of input `hparams`
        during module construction.

        .. code-block:: python

            {
                "name": "module"
            }
        """
        return {
            "name": "module"
        }

    def _build(self, *args, **kwargs):
        """Subclass must implement this method to build the logic.

        Args:
            *args: Arguments.
            **kwargs: Keyword arguments.

        Returns:
            Output Tensor(s).
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Executes the module logic defined in _build method

        Args:
            *args: Arguments of _build method.
            **kwargs: Keyword arguments of _build method.

        Returns:
            The output of _build method.
        """
        return self._template(*args, **kwargs)

    def _add_internal_trainable_variables(self):  # pylint: disable=invalid-name
        """Collects trainable variables constructured internally in this module.

        This is typically called at the end of `_build()` where all necessary
        trainable variables have been constructed.
        """
        scope_name = self.variable_scope.name
        # Escape to handle possible "." characters in the name.
        # Append a slash to the end to avoid searching scopes that have this
        # scope name as a prefix.
        scope_name = re.escape(scope_name) + "/"
        internal_trainable_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
        self._add_trainable_variable(internal_trainable_variables)

    def _add_trainable_variable(self, variable):
        """Adds a trainable variable to the trainable variable list of the
        module.

        Args:
            variable: a (list of) trainable variable(s) constructed either
                internally in the module or constructured outside but used
                inside the module.
        """
        if isinstance(variable, (list, tuple)):
            for var in variable:
                self._add_trainable_variable(var)
        else:
            if variable not in self._trainable_variables:
                self._trainable_variables.append(variable)

    @property
    def variable_scope(self):
        """The variable scope of the module.
        """
        return self._template.variable_scope

    @property
    def name(self):
        """The uniquified name of the module.
        """
        return self._unique_name

    @property
    def trainable_variables(self):
        """The list of trainable variables of the module.
        """
        if not self._built:
            raise TexarError(
                "Attempting to access trainable_variables before module %s "
                "was fully built. The module is built once it is called, "
                "e.g., with `%s(...)`" % (self.name, self.name))
        return self._trainable_variables

    @property
    def hparams(self):
        """An :class:`~texar.tf.HParams` instance. The hyperparameters
        of the module.
        """
        return self._hparams
