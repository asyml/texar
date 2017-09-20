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

    def __init__(self, hparams, default_hparams, allow_new_hparam=False):
        """
        Args:
            hparams: A dictionary of hyperparameters.
            default_hparams: A dictionary of hyperparameters with default
                values.
            allow_new_hparam: Boolean. If `False` (default), raise error if
                hyperparameter name is not in `default_hparams`. If `True`,
                new hyperparameters not in `default_hparams` are added.
        """
        if default_hparams is not None:
            parsed_hparams = self._parse(
                hparams, default_hparams, allow_new_hparam)
        else:
            parsed_hparams = self._parse(hparams, hparams)
        super(HParams, self).__setattr__('_hparams', parsed_hparams)

    @staticmethod
    def _parse(hparams, default_hparams, allow_new_hparam=False): # pylint: disable=too-many-branches
        """Parses hyperparameters.

        Replaces missing values with default values, and checks the types of
        values. Hyperparameter named "kwargs" is the arguments of a function.
        For such hyperparameters only typecheck is performed.

        Args:
            hparams: A dictionary of hyperparameters.
            default_hparams: A dictionary of hyperparameters with default
                values.
            allow_new_hparam: Boolean. If `False` (default), raise error if
                hyperparameter name is not in `default_hparams`. If `True`,
                new hyperparameters not in `default_hparams` are added.
        """
        if hparams is None and default_hparams is None:
            return None

        if hparams is None:
            return HParams._parse(default_hparams, default_hparams)

        parsed_hparams = copy.deepcopy(default_hparams)

        for name, value in hparams.items():
            if name not in default_hparams:
                if allow_new_hparam:
                    parsed_hparams[name] = HParams._parse_value(value, name)
                    continue
                else:
                    raise ValueError(
                        "Unknown hyperparameter: %s. Only the `kwargs` "
                        "hyperparameters can contain new entries undefined "
                        "in default hyperparameters." % name)

            if value is None:
                parsed_hparams[name] = \
                    HParams._parse_value(parsed_hparams[name])

            default_value = default_hparams[name]
            if default_value is None:
                parsed_hparams[name] = HParams._parse_value(value)
                continue

            # Parse recursively for param of type dictionary
            if isinstance(value, dict):
                if not isinstance(default_value, dict):
                    raise ValueError(
                        "Hyperparameter '%s' must have type %s, got %s" %
                        (name, _type_name(default_value), _type_name(value)))
                if name != "kwargs":
                    parsed_hparams[name] = HParams(
                        value, default_value, allow_new_hparam)
                else:
                    # Allow new items for function keyword args
                    parsed_hparams[name] = HParams(
                        value, default_value, allow_new_hparam=True)

                continue

            try:
                parsed_hparams[name] = type(default_value)(value)
            except TypeError:
                raise ValueError(
                    "Hyperparameter '%s' must have type %s, got %s" %
                    (name, _type_name(default_value), _type_name(value)))

        return parsed_hparams

    @staticmethod
    def _parse_value(value, name=None):
        if isinstance(value, dict) and (name is None or name != "kwargs"):
            return HParams(value, None)
        else:
            return value

    def __getattr__(self, name):
        """Retrieves the value of the hyperparameter.
        """
        if name not in self._hparams:
            # Raise AttributeError to allow copy.deepcopy
            raise AttributeError("Unknown hyperparameter: %s" % name)
        return self._hparams[name]

    def __getitem__(self, name):
        """Retrieves the value of the hyperparameter.
        """
        return self.__getattr__(name)

    def __setattr__(self, name, value):
        """Sets the value of the hyperparameter.
        """
        if name not in self._hparams:
            raise ValueError(
                "Unknown hyperparameter: %s. Only the `kwargs` "
                "hyperparameters can contain new entries undefined "
                "in default hyperparameters." % name)
        self._hparams[name] = self._parse_value(value, name)

    def items(self):
        """Returns the list of hyperparam `(name, value)` pairs
        """
        return iter(self)

    def keys(self):
        """Returns the list of hyperparam names
        """
        return self._hparams.keys()

    def __iter__(self):
        for name, value in self._hparams.items():
            yield name, value

    def __len__(self):
        return len(self._hparams)

    def __contains__(self, name):
        return name in self._hparams

    def add_hparam(self, name, value):
        """Adds a new hyperparameter.
        """
        if (name in self._hparams) or hasattr(self, name):
            raise ValueError("Hyperparameter name already exists: %s" % name)
        self._hparams[name] = self._parse_value(value, name)

    def todict(self):
        """Returns a copy of hyperparameters as a dictionary.
        """
        dict_ = copy.deepcopy(self._hparams)
        for name, value in self._hparams.items():
            if isinstance(value, HParams):
                dict_[name] = value.todict()
        return dict_

