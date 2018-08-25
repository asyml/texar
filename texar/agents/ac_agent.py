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
"""Actor-critic agent.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from texar.agents.episodic_agent_base import EpisodicAgentBase
from texar.utils import utils

# pylint: disable=too-many-instance-attributes, protected-access
# pylint: disable=too-many-arguments

__all__ = [
    "ActorCriticAgent"
]


class ActorCriticAgent(EpisodicAgentBase):
    """Actor-critic agent for episodic setting.

    An actor-critic algorithm consists of several components:

        - **Actor** is the policy to optimize. As a temporary implementation,\
        here by default we use a :class:`~texar.agents.PGAgent` instance \
        that wraps a `policy net` and provides proper interfaces to perform \
        the role of an actor.
        - **Critic** that provides learning signals to the actor. Again, as \
        a temporary implemetation, here by default we use a \
        :class:`~texar.agents.DQNAgent` instance that wraps a `Q net` and \
        provides proper interfaces to perform the role of a critic.

    Args:
        env_config: An instance of :class:`~texar.agents.EnvConfig` specifying
            action space, observation space, and reward range, etc. Use
            :func:`~texar.agents.get_gym_env_config` to create an EnvConfig
            from a gym environment.
        sess (optional): A tf session.
            Can be `None` here and set later with `agent.sess = session`.
        actor (optional): An instance of :class:`~texar.agents.PGAgent` that
            performs as actor in the algorithm.
            If not provided, an actor is created based on :attr:`hparams`.
        actor_kwargs (dict, optional): Keyword arguments for actor
            constructor. Note that the `hparams` argument for actor
            constructor is specified in the "actor_hparams" field of
            :attr:`hparams` and should not be included in `actor_kwargs`.
            Ignored if :attr:`actor` is given.
        critic (optional): An instance of :class:`~texar.agents.DQNAgent` that
            performs as critic in the algorithm.
            If not provided, a critic is created based on :attr:`hparams`.
        critic_kwargs (dict, optional): Keyword arguments for critic
            constructor. Note that the `hparams` argument for critic
            constructor is specified in the "critic_hparams" field of
            :attr:`hparams` and should not be included in `critic_kwargs`.
            Ignored if :attr:`critic` is given.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparamerters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.
    """

    def __init__(self,
                 env_config,
                 sess=None,
                 actor=None,
                 actor_kwargs=None,
                 critic=None,
                 critic_kwargs=None,
                 hparams=None):
        EpisodicAgentBase.__init__(self, env_config=env_config, hparams=hparams)

        self._sess = sess
        self._num_actions = self._env_config.action_space.high - \
                            self._env_config.action_space.low

        with tf.variable_scope(self.variable_scope):
            if actor is None:
                kwargs = utils.get_instance_kwargs(
                    actor_kwargs, self._hparams.actor_hparams)
                kwargs.update(dict(env_config=env_config, sess=sess))
                actor = utils.get_instance(
                    class_or_name=self._hparams.actor_type,
                    kwargs=kwargs,
                    module_paths=['texar.agents', 'texar.custom'])
            self._actor = actor

            if critic is None:
                kwargs = utils.get_instance_kwargs(
                    critic_kwargs, self._hparams.critic_hparams)
                kwargs.update(dict(env_config=env_config, sess=sess))
                critic = utils.get_instance(
                    class_or_name=self._hparams.critic_type,
                    kwargs=kwargs,
                    module_paths=['texar.agents', 'texar.custom'])
            self._critic = critic

            if self._actor._discount_factor != self._critic._discount_factor:
                raise ValueError('discount_factor of the actor and the critic '
                                 'must be the same.')
            self._discount_factor = self._actor._discount_factor

            self._observs = []
            self._actions = []
            self._rewards = []

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values:

        .. role:: python(code)
           :language: python

        .. code-block:: python

            {
                'actor_type': 'PGAgent',
                'actor_hparams': None,
                'critic_type': 'DQNAgent',
                'critic_hparams': None,
                'name': 'actor_critic_agent'
            }

        Here:

        "actor_type" : str or class or instance
            Actor. Can be class, its
            name or module path, or a class instance. If class name is given,
            the class must be from module :mod:`texar.agents` or
            :mod:`texar.custom`. Ignored if a `actor` is given to
            the agent constructor.

        "actor_kwargs" : dict, optional
            Hyperparameters for the actor class. With the :attr:`actor_kwargs`
            argument to the constructor, an actor is created with
            :python:`actor_class(**actor_kwargs, hparams=actor_hparams)`.

        "critic_type" : str or class or instance
            Critic. Can be class, its
            name or module path, or a class instance. If class name is given,
            the class must be from module :mod:`texar.agents` or
            :mod:`texar.custom`. Ignored if a `critic` is given to
            the agent constructor.

        "critic_kwargs" : dict, optional
            Hyperparameters for the critic class. With the :attr:`critic_kwargs`
            argument to the constructor, an critic is created with
            :python:`critic_class(**critic_kwargs, hparams=critic_hparams)`.

        "name" : str
            Name of the agent.
        """
        return {
            'actor_type': 'PGAgent',
            'actor_hparams': None,
            'critic_type': 'DQNAgent',
            'critic_hparams': None,
            'name': 'actor_critic_agent'
        }

    def _reset(self):
        self._actor._reset()
        self._critic._reset()

    def _observe(self, reward, terminal, train_policy, feed_dict):
        self._train_actor(
            observ=self._observ,
            action=self._action,
            feed_dict=feed_dict)
        self._critic._observe(reward, terminal, train_policy, feed_dict)

    def _train_actor(self, observ, action, feed_dict):
        qvalues = self._critic._qvalues_from_target(observ=observ)
        advantage = qvalues[0][action] - np.mean(qvalues)
        # TODO (bowen): should be a funciton to customize?

        feed_dict_ = {
            self._actor._observ_inputs: [observ],
            self._actor._action_inputs: [action],
            self._actor._advantage_inputs: [advantage]
        }
        feed_dict_.update(feed_dict)

        self._actor._train_policy(feed_dict=feed_dict_)

    def get_action(self, observ, feed_dict=None):
        self._observ = observ
        self._action = self._actor.get_action(observ, feed_dict=feed_dict)

        self._critic._update_observ_action(self._observ, self._action)

        return self._action

    @property
    def sess(self):
        """The tf session.
        """
        return self._sess

    @sess.setter
    def sess(self, session):
        self._sess = session
        self._actor._sess = session
        self._critic._sess = session
