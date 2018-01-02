#
"""TODO: add docs
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.hyperparams import HParams


class AgentBase(object):
    """TODO: docs
    """
    def __init__(self, hparams=None):
        self._hparams = HParams(hparams, self.default_hparams())
        self._variable_scope = None
        self.current_state = None
        self.timestep = 0

    @staticmethod
    def default_hparams():
        """TODO: docs
        """
        return {
            'name': 'agent_base'
        }

    def set_initial_state(self, observation):
        """TODO: docs
        """
        raise NotImplementedError

    def perceive(self, action, reward, is_terminal, next_state):
        """TODO: docs
        """
        raise NotImplementedError

    def get_action(self, state, action_mask=None):
        """TODO: docs
        """
        raise NotImplementedError

    @property
    def variable_scope(self):
        """The variable scope of the agent.
        """
        return self._variable_scope

    @property
    def name(self):
        """The uniquified name of the module.
        """
        # pylint: disable=protected-access
        return self.variable_scope._pure_variable_scope._name_or_scope
