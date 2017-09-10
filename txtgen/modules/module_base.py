#
"""
Base class for modules.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf

from txtgen.hyperparams import HParams


class ModuleBase(object):
    """Base class inherited by modules that create Variables and are
    configurable through hyperparameters.

    Args:
        name (string): Name of the module.
        hparams (dict, optional): Hyperparameters of the module. See
            :attr:`default_hparams` for the structure and default values.
    """

    def __init__(self, name, hparams=None):
        self.name = name
        self._template = tf.make_template(name, self._build,
                                          create_scope_now_=True)
        self._hparams = HParams(hparams, self.default_hparams())
        self._unique_name = self.variable_scope.name.split("/")[-1]
        self._trainable_variables = []
        self._built = False

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters of the module with default
        values. Used to replace the missing values of input :attr:`hparams`
        during module construction.
        """
        raise NotImplementedError

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
    def module_name(self):
        """The name of the module.
        """
        return self._unique_name

    @property
    def trainable_variables(self):
        """The list of trainable variables of the module.
        """
        if not self._built:
            raise ValueError(
                "Attempting to access trainable_variables before module %s "
                "was fully built. The module is built once it is called, "
                "e.g., with `%s(...)`" % (self.module_name, self.module_name))
        return self._trainable_variables

    @property
    def hparams(self):
        """The hyperparameters of the module.
        """
        return self._hparams
