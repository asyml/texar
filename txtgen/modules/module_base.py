#
"""
Base class for modules
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
        self._template = tf.make_template(name, self._build, create_scope_now_=True)
        self._hparams = HParams(hparams, self.default_hparams())
        self._unique_name = self._template.variable_scope.name.split("/")[-1]
        self._variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope=self.variable_scope.name)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of default hyperparameters of the module.
        Used to replace the missing values of input hyperparameters during
        initialization
        """
        raise NotImplementedError

    def _build(self, *args, **kwargs):
        """Subclass must implement this method to build the logic

        Args:
          *args: Input Tensors.
          **kwargs: Additional Python flags controlling connection.

        Returns:
          output Tensor(s)
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Executes the module logic defined in _build method

        Args:
          *args: Arguments for _build method.
          **kwargs: Keyword arguments for _build method.

        Returns:
          The output of _build method.
        """
        return self._template(*args, **kwargs)

    @property
    def variable_scope(self):
        """Returns the variable scope of the module
        """
        return self._template.variable_scope

    @property
    def module_name(self):
        """Returns the name of the module
        """
        return self._unique_name

    @property
    def variables(self):
        """Returns the list of trainable variables of the module
        """
        return self._variables

    @property
    def hparams(self):
        """Returns the hyperparameters of the module
        """
        return self._hparams
