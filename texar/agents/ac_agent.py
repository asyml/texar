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
            self.actor = actor

            if critic is None:
                kwargs = utils.get_instance_kwargs(
                    critic_kwargs, self._hparams.critic_kwargs)
                kwargs.update(dict(env_config=env_config, sess=sess))
                critic = utils.get_instance(
                    class_or_name=self._hparams.critic_type,
                    kwargs=kwargs,
                    module_paths=['texar.agents', 'texar.custom'])
            self.critic = critic

            assert self.actor._discount_factor == self.critic._discount_factor
            self._discount_factor = self.actor._discount_factor

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
        self.actor._reset()
        self.critic._reset()

    def _observe(self, observ, action, reward, terminal, next_observ,
                 train_policy, feed_dict):
        self.critic._observe(observ, action, reward, terminal, next_observ,
                 train_policy, feed_dict)

        feed_dict_ = {self.critic._observ_inputs: [next_observ]}
        feed_dict_.update(feed_dict)
        next_step_qvalues = self.critic._sess.run(
            self.critic._qnet_outputs['qvalues'], feed_dict=feed_dict_)

        action_one_hot = [0.] * self._num_actions
        action_one_hot[action] = 1.
        feed_dict_ = {
            self.critic._observ_inputs: [observ],
            self.critic._y_inputs:
                [reward + self._discount_factor * next_step_qvalues[0][action]],
            self.critic._action_inputs: [action_one_hot]
        }
        feed_dict_.update(feed_dict)
        td_errors = self.critic._sess.run(
            self.critic._td_error, feed_dict=feed_dict_)

        feed_dict_ = {
            self.actor._observ_inputs: [observ],
            self.actor._action_inputs: [action],
            self.actor._advantage_inputs: td_errors
        }
        feed_dict_.update(feed_dict)

        self.actor._train_policy(feed_dict=feed_dict_)

    def get_action(self, observ, feed_dict=None):
        return self.actor.get_action(observ, feed_dict=feed_dict)

    @property
    def sess(self):
        """The tf session.
        """
        return self._sess

    @sess.setter
    def sess(self, session):
        self._sess = session
        self.actor._sess = session
        self.critic._sess = session


