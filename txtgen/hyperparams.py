#
"""
Hyperparameter manager
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import copy


__all__ = [
    "HParams"
]

def _type_name(value):
    return type(value).__name__


class HParams(object):
    """Hyperparameters.

    Type check is performed to make sure the value types of hyperparameters in
    :attr:`hparams` are consistent with their default value types in
    :attr:`default_hparams`.
    Missing hyperparameters are set to default values. The only exception
    happens when :attr:`default_hparams` has the structure:

    .. code-block:: python

        {
            "type": "<some type>",
            "kwargs": { ... }
            # Other hyperparameters
            # ...
        }

    Here :attr:`"type"` is the name or full path to a function or a class,
    and :attr:`"kwargs"` is the arguments for the function or the
    constructor of the class.

    - The default hyperparameters in :attr:`"kwargs"` are used (for typecheck \
    and complementing missing hyperparameters) only when :attr:`"type"` \
    takes default value (i.e., missing in :attr:`hparams` or set to \
    the same value with the default). In this case :attr:`kwargs` allows to \
    contain new keys not included in :attr:`default_hparams["kwargs"]`.
    - If :attr:`"type"` is set to an other \
    value and :attr:`"kwargs"` is missing in :attr:`hparams`, \
    :attr:`"kwargs"` is set to an empty dictionary.
    - :attr:`"type"` in :attr:`hparams` is not type-checked. Typically it \
    can be a string as with the default value, or directly the function or \
    the class instance (rather than the name or path). In the latter case, \
    :attr:`"kwargs"` is typically ignored in usage.

    Args:
        hparams (dict): Hyperparameters. If `None`, all hyperparameters are
            set to default values.
        default_hparams (dict): Hyperparameters with default values. If `None`,
            Hyperparameters are fully defined by :attr:`hparams`.
        allow_new_hparam (bool): If `False` (default), :attr:`hparams` cannot
            contain hyperparameters that are not included in
            :attr:`default_hparams`, except the case as below.
    """

    def __init__(self, hparams, default_hparams, allow_new_hparam=False):
        if default_hparams is not None:
            parsed_hparams = self._parse(
                hparams, default_hparams, allow_new_hparam)
        else:
            parsed_hparams = self._parse(hparams, hparams)
        super(HParams, self).__setattr__('_hparams', parsed_hparams)

    @staticmethod
    def _parse(hparams, # pylint: disable=too-many-branches
               default_hparams,
               allow_new_hparam=False):
        """Parses hyperparameters.

        Args:
            hparams (dict): Hyperparameters. If `None`, all hyperparameters are
                set to default values.
            default_hparams (dict): Hyperparameters with default values.
                If `None`,Hyperparameters are fully defined by :attr:`hparams`.
            allow_new_hparam (bool): If `False` (default), :attr:`hparams`
                cannot contain hyperparameters that are not included in
                :attr:`default_hparams`, except the case as below.

        Return:
            A dictionary of parsed hyperparameters. Returns `None` if both
            :attr:`hparams` and :attr:`default_hparams` are `None`.

        Raises:
            ValueError: If :attr:`hparams` is not `None` and
                :attr:`default_hparams` is `None`.
            ValueError: If :attr:`default_hparams` contains "kwargs" not does
                not contains "type".
        """
        if hparams is None and default_hparams is None:
            return None

        if hparams is None:
            return HParams._parse(default_hparams, default_hparams)

        if default_hparams is None:
            raise ValueError("`default_hparams` cannot be `None` if `hparams` "
                             "is not `None`.")

        if "kwargs" in default_hparams and "type" not in default_hparams:
            raise ValueError("Ill-defined hyperparameter structure: 'kwargs' "
                             "must accompany with 'type'.")

        parsed_hparams = copy.deepcopy(default_hparams)

        # Parse recursively for params of type dictionary that are missing
        # in `hparams`.
        for name, value in default_hparams.items():
            if name not in hparams and isinstance(value, dict):
                if name == "kwargs" and "type" in hparams and \
                        hparams["type"] != default_hparams["type"]:
                    # Set params named "kwargs" to empty dictionary if "type"
                    # takes value other than default.
                    parsed_hparams[name] = HParams({}, {})
                else:
                    parsed_hparams[name] = HParams(value, value)

        # Parse hparams
        for name, value in hparams.items():
            if name not in default_hparams:
                if allow_new_hparam:
                    parsed_hparams[name] = HParams._parse_value(value, name)
                    continue
                else:
                    raise ValueError(
                        "Unknown hyperparameter: %s. Only hyperparameters "
                        "named 'kwargs' hyperparameters can contain new "
                        "entries undefined in default hyperparameters." % name)

            if value is None:
                parsed_hparams[name] = \
                    HParams._parse_value(parsed_hparams[name])

            default_value = default_hparams[name]
            if default_value is None:
                parsed_hparams[name] = HParams._parse_value(value)
                continue

            # Parse recursively for params of type dictionary.
            if isinstance(value, dict):
                if not isinstance(default_value, dict):
                    raise ValueError(
                        "Hyperparameter '%s' must have type %s, got %s" %
                        (name, _type_name(default_value), _type_name(value)))
                if name == "kwargs":
                    if "type" in hparams and \
                            hparams["type"] != default_hparams["type"]:
                        # Leave "kwargs" as-is if "type" takes value
                        # other than default.
                        parsed_hparams[name] = HParams(value, value)
                    else:
                        # Allow new hyperparameters if "type" takes default
                        # value
                        parsed_hparams[name] = HParams(
                            value, default_value, allow_new_hparam=True)
                else:
                    parsed_hparams[name] = HParams(
                        value, default_value, allow_new_hparam)
                continue

            # Do not type-check hyperparameter named "type" and accompanied
            # with "kwargs"
            if name == "type" and "kwargs" in default_hparams:
                parsed_hparams[name] = value
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
        if name == '_hparams':
            return super(HParams, self).__getattribute__('_hparams')
        if name not in self._hparams:
            # Raise AttributeError to allow copy.deepcopy, etc
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

    def get(self, name, default=None):
        """Returns the hyperparameter value for the given name. If name is not
        available then returns :attr:`default`.

        Args:
            name (str): the name of hyperparameter.
            default: the value to be returned in case name does not exist.
        """
        try:
            return self.__getattr__(name)
        except AttributeError:
            return default

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

