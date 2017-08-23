#
"""
Hyperparameter manager
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import copy


def _type_name(value):
    return type(value).__name__


class HParams(object):
    """hyperparameters to configure modules
    """

    def __init__(self, hparams, default_hparams):
        """
        Args:
          hparams: A dictionary of hyperparameters.
          default_hparams: A dictionary of hyperparameters with default values.
        """
        self._hparams = self._parse(hparams, default_hparams)

    @staticmethod
    def _parse(hparams, default_hparams):
        """Parses hyperparameters.

        Replaces missing values with default values, and checks the types of
        values. Hyperparameter named "kwargs" is the arguments of a function.
        For such hyperparameters only typecheck is performed.

        Args:
          hparams: A dictionary of hyperparameters.
          default_hparams: A dictionary of hyperparameters with default values.
        """
        parsed_hparams = copy.deepcopy(default_hparams)

        if hparams is None:
            return parsed_hparams

        for name, value in hparams.items():
            if name not in default_hparams:
                raise ValueError("Unknown hyperparameter %s", name)
            default_value = default_hparams[name]
            if default_value is None:
                parsed_hparams[name] = value
            # parse recursively for param of type dictionary
            if isinstance(value, dict):
                if not isinstance(default_value, dict):
                    raise ValueError(
                        "Hyperparameter must have type %s, %s given: %s" %
                        (_type_name(default_value), _type_name(value), name))
                if default_value and name != "kwargs":
                    # default_value is not empty and is not function arguments
                    value = HParams._parse(value, default_value)
            if value is None:
                continue
            try:
                parsed_hparams[name] = type(default_value)(value)
            except TypeError:
                raise ValueError(
                    "Hyperparameter should have type %s, %s given: %s" %
                    (_type_name(default_value), _type_name(value), name))

        return parsed_hparams

    def __getattr__(self, name):
        """Retrieves the value of the hyperparameter
        """
        if name not in self._hparams:
            raise ValueError("Unknown hyperparameter %s", name)
        return self._hparams[name]

    def __setattr__(self, name, value):
        """Sets the value of the hyperparameter
        """
        if name not in self._hparams:
            raise ValueError("Unknown hyperparameter %s", name)
        self._hparams[name] = value

    def add_hparam(self, name, value):
        """Adds a new hyperparameter
        """
        if (name in self._hparams) or hasattr(self, name):
            raise ValueError("Hyperparameter name is reserved: %s", name)
        self._hparams[name] = value

    @property
    def dict(self):
        """Returns the dictionary of hyperparameters
        """
        return self._hparams
