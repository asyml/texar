#
"""
Base class for episodic reinforcement learning agents.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.agents.agent_base import AgentBase

# pylint: disable=too-many-instance-attributes

class EpisodicAgentBase(AgentBase):
    """
    Base class inherited by episodic RL agents.

    Args:
        TODO
    """
    def __init__(self, env_config, hparams=None):
        AgentBase.__init__(self, hparams)

        self._env_config = env_config

        self._reset_tmplt_fn = tf.make_template(
            "{}_reset".format(self.name), self._reset)
        self._observe_tmplt_fn = tf.make_template(
            "{}_observe".format(self.name), self._observe)
        self._get_action_tmplt_fn = tf.make_template(
            "{}_get_action".format(self.name), self._get_action)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        TODO
        """
        return {
            'name': 'agent'
        }

    def reset(self):
        """Resets the states to begin new episodes.
        """
        self._reset_tmplt_fn()

    def _reset(self):
        raise NotImplementedError

    def observe(self, observ, action, reward, terminal, next_observ, train_policy=True, feed_dict=None):
        """Observes experience from environment.

        Args:
        """
        return self._observe_tmplt_fn(
            observ, action, reward, terminal, next_observ, train_policy, feed_dict)

    def _observe(self, observ, action, reward, terminal, next_observ, train_policy, feed_dict):
        raise NotImplementedError

    def get_action(self, observ, feed_dict=None):
        """Gets action according to observation.

        Args:

        Returns:
        """
        return self._get_action_tmplt_fn(observ, feed_dict)

    def _get_action(self, observ, feed_dict):
        raise NotImplementedError

