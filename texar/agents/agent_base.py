#
"""
Base class for RL agents.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.hyperparams import HParams


class AgentBase(object):
    """
    Base class inherited by RL agents.

    Args:
        TODO
    """
    def __init__(self, hparams=None):
        self._hparams = HParams(hparams, self.default_hparams())
        self._variable_scope = None
        self.current_state = None
        self.timestep = 0

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        return {
            'name': 'agent_base'
        }

    def set_initial_state(self, observation):
        """Resets the current state.

        Args:
            observation: observation in the beginning
        """
        raise NotImplementedError

    def perceive(self, action, reward, is_terminal, next_observation):
        """Perceives from environment.

        Args:
            action: A one-hot vector indicate the action
            reward: A number indicate the reward
            is_terminal: True iff it is a terminal state
            next_observation: New Observation from environment
        """
        raise NotImplementedError

    def get_action(self, state=None, action_mask=None):
        """Get Action according to state and action_mask

        Args:
            state(numpy.array): assign a state if it is not 'None', otherwise it
                is current state by default
            action_mask(list): A List of True or False, indicate this time each
                action can be take or not, if it is 'None', then all the actions
                can be take.

        Returns:
            list: The possibility of taking each action.
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
