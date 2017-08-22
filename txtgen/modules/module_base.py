#
"""
Base class for modules
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf

from txtgen.hyperparams import HParams


class ModuleBase(object):
    """Base class inherited by modules that create Variables and are
    configurable through hyperparameters
    """

    def __init__(self, name, hparams=None):
        """Initialize the module.

        Args:
            name: Name of the module
            hparams: A dictionary of hyperparameters
        """
        self.name = name
        self._template = tf.make_template(name, self._build,
                                          create_scope_now_=True)
        self._hparams = HParams(hparams, self.default_hparams())
        self._unique_name = self._template.variable_scope.name.split("/")[-1]
        self._trainable_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.variable_scope.name)


    def _build(self, *args, **kwargs):
        """Subclass must implement this method to build the logic.

        Args:
          *args: Arguments.
          **kwargs: Keyword arguments.

        Returns:
          output Tensor(s).
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
        if isinstance(variable, list):
            for var in variable:
                if var not in self.trainable_variables:
                    self._trainable_variables.append(var)
        else:
            if variable not in self.trainable_variables:
                self._trainable_variables.append(variable)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of default hyperparameters of the module.
        Used to replace the missing values of input hyperparameters during
        initialization.
        """
        raise NotImplementedError

    @property
    def variable_scope(self):
        """Returns the variable scope of the module.
        """
        return self._template.variable_scope

    @property
    def module_name(self):
        """Returns the name of the module.
        """
        return self._unique_name

    @property
    def trainable_variables(self):
        """Returns the list of trainable variables of the module.
        """
        return self._trainable_variables

    @property
    def hparams(self):
        """Returns the hyperparameters of the module.
        """
        return self._hparams
