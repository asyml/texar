""" Actor-Critic Agent
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

from texar.agents.agent_base import AgentBase
from texar.core import optimization as opt
from texar.core import get_instance
from texar.losses.pg_losses import pg_loss
from texar.losses.dqn_losses import l2_loss


class ACAgent(AgentBase): #pylint: disable=too-many-instance-attributes
    """
    Actor-Critic Agent
    (online & using td_error as advantage & no pre-train temporarily)
    """
    def __init__(self, actions, state_shape, hparams=None):
        AgentBase.__init__(self, actions, state_shape, hparams)

        self.discount_factor = self._hparams.discount_factor

        self.actor = get_instance(
            self._hparams.actor_network.type,
            {"hparams": self._hparams.actor_network.hparams},
            module_paths=['texar.modules', 'texar.custom']
        )
        with tf.variable_scope(self.actor.variable_scope):
            self.actor_state_inputs = tf.placeholder(
                dtype=tf.float64, shape=[None, ] + list(state_shape))
            self.actor_action_inputs = tf.placeholder(
                dtype=tf.int32, shape=[None, ])
            self.actor_advantage_inputs = tf.placeholder(
                dtype=tf.float64, shape=[None, ])
            self.actor_outputs = self.actor(self.actor_state_inputs)
            self.actor_policy = tf.nn.softmax(self.actor_outputs)
            self.actor_loss = self._hparams.actor_trainer.loss_fn(
                outputs=self.actor_outputs,
                action_inputs=self.actor_action_inputs,
                advantages=self.actor_advantage_inputs
            )
            self.actor_trainer = opt.get_train_op(
                loss=self.actor_loss,
                variables=None,
                hparams=self._hparams.actor_trainer.optimization_hparams
            )
        self.critic = get_instance(
            self._hparams.critic_network.type,
            {"hparams": self._hparams.critic_network.hparams},
            module_paths=['texar.modules', 'texar.custom']
        )
        with tf.variable_scope(self.critic.variable_scope):
            self.critic_state_inputs = tf.placeholder(
                dtype=tf.float64, shape=[None, ] + list(state_shape))
            self.critic_y_inputs = tf.placeholder(
                dtype=tf.float64, shape=[None, ])
            self.critic_qvalue = self.critic(self.critic_state_inputs)
            self.td_error = self.critic_y_inputs - self.critic_qvalue
            self.critic_loss = tf.reduce_sum(self.td_error ** 2)
            self.critic_trainer = opt.get_train_op(
                loss=self.critic_loss,
                variables=None,
                hparams=self._hparams.critic_trainer.optimization_hparams
            )
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    @staticmethod
    def default_hparams():
        return {
            'name': 'actor-critic agent',
            'discount_factor': 0.99,
            'actor_network': {
                'type': 'PGNet',
                'hparams': None
            },
            'actor_trainer': {
                'loss_fn': pg_loss,
                'optimization_hparams': opt.default_optimization_hparams()
            },
            'critic_network': {
                'type': 'SimpleQNet',
                'hparams': None
            },
            'critic_trainer': {
                'loss_fn': l2_loss,
                'optimization_hparams': opt.default_optimization_hparams()
            }
        }

    def set_initial_state(self, observation):
        self.current_state = np.array(observation)

    def train_critic(self, state, reward, next_observation):
        """
        Train the Critic
        :param state:
        :param reward:
        :param next_observation:
        :return: The td_error which will be given to Actor
        """
        next_step_qvalue = \
            self.sess.run(self.critic_qvalue, feed_dict={
                self.critic_state_inputs: [next_observation, ]})[0][0]
        y_input = reward + self.discount_factor * next_step_qvalue

        self.sess.run(self.critic_trainer, feed_dict={
            self.critic_state_inputs: [state, ],
            self.critic_y_inputs: [y_input, ]
        })
        return self.sess.run(self.td_error, feed_dict={
            self.critic_state_inputs: [state, ],
            self.critic_y_inputs: [y_input, ]
        })

    def train_actor(self, state, action_id, advantage):
        """
        Train the Actor
        :param state:
        :param action_id:
        :param advantage:
        :return:
        """
        self.sess.run(self.actor_trainer, feed_dict={
            self.actor_state_inputs: [state, ],
            self.actor_action_inputs: [action_id, ],
            self.actor_advantage_inputs: [advantage, ]
        })

    def perceive(self, action_id, reward, is_terminal, next_observation):
        advantage = self.train_critic(
            self.current_state, reward, next_observation)[0][0]
        self.train_actor(self.current_state, action_id, advantage)

        self.current_state = next_observation

    def get_action(self, state=None, action_mask=None):
        if state is None:
            state = self.current_state
        if action_mask is None:
            action_mask = [True, ] * self.actions

        probs = self.sess.run(self.actor_policy,
                              feed_dict={self.actor_state_inputs: [state, ]})[0]

        prob_sum = 0.
        for i in range(self.actions):
            if action_mask[i] is True:
                prob_sum += probs[i]
        for i in range(self.actions):
            if action_mask[i] is True:
                probs[i] /= prob_sum
            else:
                probs[i] = 0.

        return np.random.choice(self.actions, p=probs)
