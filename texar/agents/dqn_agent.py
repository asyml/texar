"""Deep Q learning Agent.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np

import tensorflow as tf

import texar as tx
from texar.agents.episodic_agent_base import EpisodicAgentBase
from texar.utils import utils
from texar.core import optimization as opt

# pylint: disable=too-many-instance-attributes, too-many-arguments
# pylint: disable=invalid-name

__all__ = [
    "DQNAgent"
]


# TODO(zhiting): only support discrete actions?
class DQNAgent(EpisodicAgentBase):
    """Deep Q learning agent.
    """
    def __init__(self,
                 env_config,
                 sess=None,
                 qnet=None,
                 target=None,
                 qnet_kwargs=None,
                 qnet_caller_kwargs=None,
                 replay_memory=None,
                 replay_memory_kwargs=None,
                 exploration=None,
                 exploration_kwargs=None,
                 hparams=None):
        EpisodicAgentBase.__init__(self, env_config, hparams)

        self._sess = sess
        self._cold_start_steps = self._hparams.cold_start_steps
        self._sample_batch_size = self._hparams.sample_batch_size
        self._update_period = self._hparams.update_period
        self._discount_factor = self._hparams.discount_factor
        self._update_type = self._hparams.update_type
        self._num_actions = self._env_config.action_space.high - \
                            self._env_config.action_space.low

        with tf.variable_scope(self.variable_scope):
            if qnet is None:
                kwargs = utils.get_instance_kwargs(
                    qnet_kwargs, self._hparams.qnet_hparams)
                qnet = utils.check_or_get_instance(
                    ins_or_class_or_name=self._hparams.qnet_type,
                    kwargs=kwargs,
                    module_paths=['texar.modules', 'texar.custom'])
                target = utils.check_or_get_instance(
                    ins_or_class_or_name=self._hparams.qnet_type,
                    kwargs=kwargs,
                    module_paths=['texar.modules', 'texar.custom'])
            self._qnet = qnet
            self._target = target
            self._qnet_caller_kwargs = qnet_caller_kwargs or {}

            if replay_memory is None:
                kwargs = utils.get_instance_kwargs(
                    replay_memory_kwargs, self._hparams.replay_memory_hparams)
                replay_memory = utils.check_or_get_instance(
                    ins_or_class_or_name=self._hparams.replay_memory_type,
                    kwargs=kwargs,
                    module_paths=['texar.core', 'texar.custom'])
            self._replay_memory = replay_memory

            if exploration is None:
                kwargs = utils.get_instance_kwargs(
                    exploration_kwargs, self._hparams.exploration_hparams)
                exploration = utils.check_or_get_instance(
                    ins_or_class_or_name=self._hparams.exploration_type,
                    kwargs=kwargs,
                    module_paths=['texar.core', 'texar.custom'])
            self._exploration = exploration

        self._build_graph()
        self.timestep = 0

    @staticmethod
    def default_hparams():
        return {
            'qnet_type': 'CategoricalQNet',
            'qnet_hparams': None,
            'replay_memory_type': 'DequeReplayMemory',
            'replay_memory_hparams': None,
            'exploration_type': 'EpsilonLinearDecayExploration',
            'exploration_hparams': None,
            'optimization': opt.default_optimization_hparams(),
            'update_type': 'copy',
            'cold_start_steps': 100,
            'sample_batch_size': 32,
            'update_period': 100,
            'discount_factor': 0.95,
            'name': 'dqn_agent'
        }

    def _build_graph(self):
        with tf.variable_scope(self.variable_scope):
            self._observ_inputs = tf.placeholder(
                dtype=self._env_config.observ_dtype,
                shape=[None, ] + list(self._env_config.observ_shape),
                name='observ_inputs')
            self._action_inputs = tf.placeholder(
                dtype=self._env_config.action_dtype,
                shape=[None, self._num_actions],
                name='action_inputs')
            # TODO(zhiting): the name `y` is not readable.
            self._y_inputs = tf.placeholder(
                dtype=tf.float32,
                shape=[None, ],
                name='y_inputs')

            self._qnet_outputs = self._get_qnet_outputs(self._observ_inputs)
            self._target_outputs = self._get_target_outputs(self._observ_inputs)
            self._td_error = self._get_td_error(
                qnet_qvalues=self._qnet_outputs['qvalues'],
                actions=self._action_inputs,
                y=self._y_inputs)
            self._train_op = self._get_train_op()

            if self._update_type == 'copy':
                self._update_op = self._get_copy_update_op()
            elif self._update_type == 'tau':
                self._update_op = self._get_tau_update_op()

    def _get_qnet_outputs(self, state_inputs):
        return self._qnet(inputs=state_inputs, **self._qnet_caller_kwargs)

    def _get_target_outputs(self, state_inputs):
        return self._target(inputs=state_inputs, **self._qnet_caller_kwargs)

    def _get_td_error(self, qnet_qvalues, actions, y):
        return y - tf.reduce_sum(qnet_qvalues * tf.to_float(actions), axis=1)

    def _get_train_op(self):
        train_op = opt.get_train_op(
            loss=tf.reduce_sum(self._td_error ** 2),
            variables=self._qnet.trainable_variables,
            hparams=self._hparams.optimization.todict())
        return train_op

    def _get_copy_update_op(self):
        op = []
        for i in range(len(self._qnet.trainable_variables)):
            op.append(tf.assign(ref=self._target.trainable_variables[i],
                                value=self._qnet.trainable_variables[i]))
        return op

    def _get_tau_update_op(self):
        tau = 1. / self._update_period
        op = []
        for i in range(len(self._qnet.trainable_variables)):
            op.append(tf.assign(
                ref=self._target.trainable_variables[i],
                value=(1. - tau) * self._target.trainable_variables[i] +
                      tau * self._qnet.trainable_variables[i]))
        return op

    def _observe(self, observ, action, reward, terminal, next_observ,
                 train_policy, feed_dict):
        action_one_hot = [0.] * self._num_actions
        action_one_hot[action] = 1.

        self._replay_memory.add(dict(
            observ=observ,
            action=action_one_hot,
            reward=reward,
            terminal=terminal,
            next_observ=next_observ))
        self.timestep += 1

        if self.timestep > self._cold_start_steps and train_policy:
            self._train_qnet(feed_dict)

    def _train_qnet(self, feed_dict):
        minibatch = self._replay_memory.get(self._sample_batch_size)
        observ_batch = np.array([data['observ'] for data in minibatch])
        action_batch = np.array([data['action'] for data in minibatch])
        reward_batch = np.array([data['reward'] for data in minibatch])
        terminal_batch = np.array([data['terminal'] for data in minibatch])
        next_observ_batch = \
            np.array([data['next_observ'] for data in minibatch])

        target_qvalue = self._sess.run(
            self._target_outputs['qvalues'], feed_dict={
                self._observ_inputs: next_observ_batch,
                tx.global_mode(): tf.estimator.ModeKeys.PREDICT})

        y_batch = reward_batch
        for i in range(self._sample_batch_size):
            if not terminal_batch[i]:
                y_batch[i] += self._discount_factor * np.max(target_qvalue[i])

        feed_dict_= {
            self._observ_inputs: observ_batch,
            self._y_inputs: y_batch,
            self._action_inputs: action_batch
        }
        feed_dict_.update(feed_dict or {})

        self._sess.run(self._train_op, feed_dict=feed_dict_)

        self.update_target(feed_dict)

    def update_target(self, feed_dict):
        if self._update_type == 'tau' or (
                self._update_type == 'copy' and
                self.timestep % self._update_period == 0):
            self._sess.run(self._update_op, feed_dict=feed_dict)

    def _get_action(self, observ, feed_dict=None):
        qvalue = self._sess.run(
            self._qnet_outputs['qvalues'],
            feed_dict={self._observ_inputs: np.array([observ]),
                       tx.global_mode(): tf.estimator.ModeKeys.PREDICT})

        action = np.zeros(shape=self._num_actions)
        if random.random() < self._exploration.get_epsilon(self.timestep):
            action_id = random.randrange(self._num_actions)
        else:
            action_id = np.argmax(qvalue)
        action[action_id] = 1.0

        return action_id

    def _reset(self):
        pass

    @property
    def sess(self):
        """The tf session.
        """
        return self._sess

    @sess.setter
    def sess(self, session):
        self._sess = session
