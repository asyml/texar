""" Policy Gradient Agent
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=too-many-instance-attributes

import numpy as np

import tensorflow as tf

from texar.agents.agent_base import AgentBase
from texar.core import optimization as opt
from texar.utils.utils import get_instance
from texar.losses.pg_losses import pg_loss


class PGAgent(AgentBase):
    """
    Policy Gradient Agent
    """
    def __init__(self, actions, state_shape, hparams=None):
        AgentBase.__init__(self, actions, state_shape, hparams=hparams)
        self.discount_factor = self._hparams.discount_factor

        self.network = get_instance(
            self._hparams.network.type,
            {"hparams": self._hparams.network.hparams},
            module_paths=['texar.modules', 'texar.custom']
        )

        with tf.variable_scope(self.network.variable_scope):
            self.state_input = tf.placeholder(
                dtype=tf.float64, shape=[None, ] + list(state_shape))

            self.action_inputs = tf.placeholder(dtype=tf.int32, shape=[None, ])

            self.qvalues = tf.placeholder(
                dtype=tf.float64, shape=[None, ])

            self.outputs = self.network(self.state_input)
            self.probs = tf.nn.softmax(self.outputs)

            self.loss = self._hparams.trainer.loss_fn(
                outputs=self.outputs,
                action_inputs=self.action_inputs,
                advantages=self.qvalues
            )
            self.trainer = opt.get_train_op(
                loss=self.loss,
                variables=None,
                hparams=self._hparams.trainer.optimization_hparams
            )

        self.record = list()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    @staticmethod
    def default_hparams():
        return {
            'name': 'pg_agent',
            'discount_factor': 0.95,
            'network': {
                'type': 'PGNet',
                'hparams': None
            },
            'trainer': {
                'loss_fn': pg_loss,
                'optimization_hparams': opt.default_optimization_hparams(),
            }
        }

    def set_initial_state(self, observation):
        self.current_state = np.array(observation)

    def perceive(self, action_id, reward, is_terminal, next_observation):
        self.record.append({
            'state': self.current_state,
            'action': action_id,
            'reward': reward
        })

        if is_terminal:
            self.train_network()
            self.record = list()

        self.timestep += 1
        self.current_state = np.array(next_observation)

    def train_network(self):
        """
        Train The Network
        :return:
        """
        qvalues = list()
        action_inputs = list()
        state_input = list()
        for data in self.record:
            state_input.append(data['state'])
            action_inputs.append(data['action'])
            qvalues.append(data['reward'])
        for i in range(len(qvalues) - 2, -1, -1):
            qvalues[i] += self.discount_factor * qvalues[i + 1]

        t_mean = np.mean(qvalues)
        t_std = np.std(qvalues)
        for i, value in enumerate(qvalues):
            qvalues[i] = (qvalues[i] - t_mean) / t_std

        self.sess.run(self.trainer, feed_dict={
            self.state_input: state_input,
            self.action_inputs: action_inputs,
            self.qvalues: qvalues
        })

    def get_action(self, state=None, action_mask=None):
        if state is None:
            state = self.current_state
        if action_mask is None:
            action_mask = [True, ] * self.actions

        probs = self.sess.run(self.probs,
                              feed_dict={self.state_input: [state, ]})[0]

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
