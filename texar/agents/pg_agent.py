""" Policy Gradient Agent
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=too-many-instance-attributes

import numpy as np

import tensorflow as tf

from texar.agents.agent_base import AgentBase
from texar.utils import utils
from texar.core import optimization as opt
from texar.losses.pg_losses import pg_loss


class PGAgent(AgentBase):
    """Policy Gradient Agent.
    """
    def __init__(self, env_config, policy=None, sess=None, hparams=None):
        AgentBase.__init__(env_config, hparams)

        self._sess = sess

        self._policy = policy
        if policy is None:
            raise NotImplementedError

        self._observs = []
        self._actions = []
        self._logits = []
        self._rewards = []
        self._current_timestep = 0

        self._build_graph()

    def _build_graph(self):
        with tf.variable_scope(self.variable_scope):
            self._observ_inputs = tf.placeholder(
                dtype=self._env_config.observ_dtype,
                shape=[None, ] + list(self._env_config.observ_shape),
                name='observ_inputs')
            self._action_inputs = tf.placeholder(
                dtype=self._env_config.action_dtype,
                shape=[None, ] + list(self._env_config.action_shape),
                name='action_inputs')
            self._qvalue_inputs = tf.placeholder(
                dtype=tf.float32,
                shape=[None, ],
                name='qvalue_inputs')



            #self.outputs = self.network(self.state_input)
            #self.loss = self._hparams.trainer.loss_fn(
            #    outputs=self.outputs,
            #    action_inputs=self.action_inputs,
            #    advantages=self.qvalues
            #)
            #self.trainer = opt.get_train_op(
            #    loss=self.loss,
            #    variables=None,
            #    hparams=self._hparams.trainer.optimization_hparams
            #)

    def _get_pg_loss(self):
        # TODO(zhiting): add mode
        outputs = self._policy(self._observ_inputs)


    @staticmethod
    def default_hparams():
        return {
            'name': 'pg_agent',
            'discount_factor': 0.95,
            'max_timesteps': 100,
            'network': {
                'type': 'PGNet',
                'hparams': None
            },
            'trainer': {
                'loss_fn': pg_loss,
                'optimization_hparams': opt.default_optimization_hparams(),
            }
        }

    def _reset(self):
        pass

    def _get_action(self, observ, mode, feed_dict=None):
        output = self._policy(observ=observ, mode=mode)

        action = output['action']
        if self._sess is not None:
            fetches = dict(action=action, logit=logit)
            val = self._sess.run(fetches, feed_dict=feed_dict)
            action = val['action']
            logit = val['logit']

        self._observs.append(observ)
        self._actions.append(action)
        self._logits.append(logit)

        return action

    def _observe(self, reward, terminal, mode):
        self._rewards.append(reward)

        if terminal:
            tf.cond(
                utils.is_train_mode(mode),
                self.train_policy, tf.no_op)

    def train_policy(self):
        """Updates the policy.

        Args:
            TODO

        Returns:
        """
        pass
        #discount_factor = self._hparams.discount_factor
        #qvalues = list(self._rewards)
        #for i in range(len(qvalues) - 2, -1, -1):
        #    qvalues[i] += discount_factor * qvalues[i + 1]

        #q_mean = np.mean(qvalues)
        #q_std = np.std(qvalues)
        #qvalues = [(q - q_mean) / q_std for q in qvalues]

        #self.sess.run(self.trainer, feed_dict={
        #    self.state_input: state_input,
        #    self.action_inputs: action_inputs,
        #    self.qvalues: qvalues
        #})

