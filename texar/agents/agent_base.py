#
"""
Base class for reinforcement learning agents.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.hyperparams import HParams
from texar.utils.variables import get_unique_named_variable_scope

# pylint: disable=too-many-instance-attributes

__all__ = [
    "AgentBase"
]

class AgentBase(object):
    """
    Base class inherited by RL agents.

    Args:
        TODO
    """
    def __init__(self, hparams=None):
        self._hparams = HParams(hparams, self.default_hparams())

        name = self._hparams.name
        self._variable_scope = get_unique_named_variable_scope(name)
        self._unique_name = self._variable_scope.name.split("/")[-1]

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        TODO
        """
        return {
            'name': 'agent'
        }

    @property
    def variable_scope(self):
        """The variable scope of the agent.
        """
        return self._variable_scope

    @property
    def name(self):
        """The name of the module (not uniquified).
        """
        return self._unique_name

    @property
    def hparams(self):
        """A :class:`~texar.hyperparams.HParams` instance. The hyperparameters
        of the module.
        """
        return self._hparams
