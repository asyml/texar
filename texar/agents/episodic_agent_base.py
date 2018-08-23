# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
    """Base class inherited by episodic RL agents.

    An agent is a wrapper of the **training process** that trains a model
    with RL algorithms. Agent itself does not create new trainable variables.

    An episodic RL agent typically provides 3 interfaces, namely, :meth:`reset`,
    :meth:`get_action` and :meth:`observe`, and is used as the following
    example.

    Example:

        .. code-block:: python

            env = SomeEnvironment(...)
            agent = PGAgent(...)

            while True:
                # Starts one episode
                agent.reset()
                observ = env.reset()
                while True:
                    action = agent.get_action(observ)
                    next_observ, reward, terminal = env.step(action)
                    agent.observe(reward, terminal)
                    observ = next_observ
                    if terminal:
                        break

    Args:
        env_config: An instance of :class:`~texar.agents.EnvConfig` specifying
            action space, observation space, and reward range, etc. Use
            :func:`~texar.agents.get_gym_env_config` to create an EnvConfig
            from a gym environment.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.
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

        .. code-block:: python

            {
                "name": "agent"
            }
        """
        return {
            'name': 'agent'
        }

    def reset(self):
        """Resets the states to begin new episode.
        """
        self._reset_tmplt_fn()

    def _reset(self):
        raise NotImplementedError

    def observe(self, reward, terminal, train_policy=True, feed_dict=None):
        """Observes experience from environment.

        Args:
            reward: Reward of the action. The configuration (e.g., shape) of
                the reward is defined in :attr:`env_config`.
            terminal (bool): Whether the episode is terminated.
            train_policy (bool): Wether to update the policy for this step.
            feed_dict (dict, optional): Any stuffs fed to running the training
                operator.
        """
        return self._observe_tmplt_fn(reward, terminal, train_policy, feed_dict)

    def _observe(self, reward, terminal, train_policy, feed_dict):
        raise NotImplementedError

    def get_action(self, observ, feed_dict=None):
        """Gets action according to observation.

        Args:
            observ: Observation from the environment.

        Returns:
            action from the policy.
        """
        return self._get_action_tmplt_fn(observ, feed_dict)

    def _get_action(self, observ, feed_dict):
        raise NotImplementedError

    @property
    def env_config(self):
        """Environment configuration.
        """
        return self._env_config
