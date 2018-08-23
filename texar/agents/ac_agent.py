from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.agents.episodic_agent_base import EpisodicAgentBase
from texar.agents import DQNAgent, PGAgent
from texar.utils import utils

__all__ = [
    "ActorCriticAgent"
]


class ActorCriticAgent(EpisodicAgentBase):
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
                    actor_kwargs, self._hparams.actor_kwargs)
                kwargs.update(dict(env_config=env_config, sess=sess))
                actor = utils.get_instance(
                    class_or_name=self._hparams.actor_type,
                    kwargs=kwargs,
                    module_paths=['texar.agents', 'texar.custom'])
            self._actor = actor

            if critic is None:
                kwargs = utils.get_instance_kwargs(
                    critic_kwargs, self._hparams.critic_kwargs)
                kwargs.update(dict(env_config=env_config, sess=sess))
                critic = utils.get_instance(
                    class_or_name=self._hparams.critic_type,
                    kwargs=kwargs,
                    module_paths=['texar.agents', 'texar.custom'])
            self._critic = critic

            assert self._actor._discount_factor == self._critic._discount_factor
            self._discount_factor = self._actor._discount_factor

            self._observs = []
            self._actions = []
            self._rewards = []

    @staticmethod
    def default_hparams():
        return {
            'actor_type': 'PGAgent',
            'actor_kwargs': None,
            'actor_hparams': PGAgent.default_hparams(),
            'critic_type': 'DQNAgent',
            'critic_kwargs': None,
            'critic_hparams': DQNAgent.default_hparams(),
            'name': 'actor_critic_agents'
        }

    def _reset(self):
        self._actor._reset()
        self._critic._reset()

        self._observs = []
        self._actions = []
        self._rewards = []

    def _observe(self, reward, terminal, train_policy, feed_dict):
        self._rewards.append(reward)
        if len(self._observs) >= 2:
            self._train_actor(
                observ=self._observs[-2],
                action=self._actions[-1],
                reward=self._rewards[-1],
                next_observ=self._observs[-1],
                feed_dict=feed_dict)
        self._critic._observe(reward, terminal, train_policy, feed_dict)

    def _train_actor(self, observ, action, reward, next_observ, feed_dict):
        feed_dict_ = {self._critic._observ_inputs: [next_observ]}
        feed_dict_.update(feed_dict)
        next_step_qvalues = self._critic._qvalues_from_qnet(next_observ)

        action_one_hot = [0.] * self._num_actions
        action_one_hot[action] = 1.
        feed_dict_ = {
            self._critic._observ_inputs: [observ],
            self._critic._y_inputs:
                [reward + self._discount_factor * next_step_qvalues[0][action]],
            self._critic._action_inputs: [action_one_hot]
        }
        feed_dict_.update(feed_dict)
        td_errors = self._critic._sess.run(
            self._critic._td_error, feed_dict=feed_dict_)

        feed_dict_ = {
            self._actor._observ_inputs: [observ],
            self._actor._action_inputs: [action],
            self._actor._advantage_inputs: td_errors
        }
        feed_dict_.update(feed_dict)

        self._actor._train_policy(feed_dict=feed_dict_)

    def get_action(self, observ, feed_dict=None):
        self._observs.append(observ)
        self._actions.append(self._actor.get_action(
            observ, feed_dict=feed_dict))

        self._critic._update_observ_action(
            self._observs[-1], self._actions[-1])

        return self._actions[-1]

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

