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


class DQNAgent(EpisodicAgentBase):
    """Deep Q learning agent for episodic setting.

    A Q learning algorithm consists of several components:

        - A **Q-net** takes in a state and returns Q-value for action sampling.\
        See :class:`~texar.modules.CategoricalQNet` for an example Q-net class\
        and required interface.
        - A **replay memory** manages past experience for Q-net updates. See\
        :class:`~texar.core.DequeReplayMemory` for an example replay memory\
        class and required interface.
        - An **exploration** that specifies the exploration strategy used\
        to train the Q-net. See\
        :class:`~texar.core.EpsilonLinearDecayExploration` for an example\
        class and required interface.

    Args:
        env_config: An instance of :class:`~texar.agents.EnvConfig` specifying
            action space, observation space, and reward range, etc. Use
            :func:`~texar.agents.get_gym_env_config` to create an EnvConfig
            from a gym environment.
        sess (optional): A tf session.
            Can be `None` here and set later with `agent.sess = session`.
        qnet (optional): A Q network that predicts Q values given states.
            If not given, a Q network is created based on :attr:`hparams`.
        target (optional): A target network to compute target Q values.
        qnet_kwargs (dict, optional): Keyword arguments for qnet
            constructor. Note that the `hparams` argument for network
            constructor is specified in the "policy_hparams" field of
            :attr:`hparams` and should not be included in `policy_kwargs`.
            Ignored if :attr:`qnet` is given.
        qnet_caller_kwargs (dict, optional): Keyword arguments for
            calling `qnet` to get Q values. The `qnet` is called with
            :python:`outputs=qnet(inputs=observation, **qnet_caller_kwargs)`
        replay_memory (optional): A replay memory instance.
            If not given, a replay memory is created based on :attr:`hparams`.
        replay_memory_kwargs (dict, optional): Keyword arguments for
            replay_memory constructor.
            Ignored if :attr:`replay_memory` is given.
        exploration (optional): An exploration instance used in the algorithm.
            If not given, an exploration instance is created based on
            :attr:`hparams`.
        exploration_kwargs (dict, optional): Keyword arguments for exploration
            class constructor. Ignored if :attr:`exploration` is given.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparamerters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.
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
        self._target_update_strategy = self._hparams.target_update_strategy
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

        self._observ = None
        self._action = None
        self._timestep = 0

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values:

        .. role:: python(code)
           :language: python

        .. code-block:: python

            {
                'qnet_type': 'CategoricalQNet',
                'qnet_hparams': None,
                'replay_memory_type': 'DequeReplayMemory',
                'replay_memory_hparams': None,
                'exploration_type': 'EpsilonLinearDecayExploration',
                'exploration_hparams': None,
                'optimization': opt.default_optimization_hparams(),
                'target_update_strategy': 'copy',
                'cold_start_steps': 100,
                'sample_batch_size': 32,
                'update_period': 100,
                'discount_factor': 0.95,
                'name': 'dqn_agent'
            }

        Here:

        "qnet_type" : str or class or instance
            Q-value net. Can be class, its
            name or module path, or a class instance. If class name is given,
            the class must be from module :mod:`texar.modules` or
            :mod:`texar.custom`. Ignored if a `qnet` is given to
            the agent constructor.

        "qnet_hparams" : dict, optional
            Hyperparameters for the Q net. With the :attr:`qnet_kwargs`
            argument to the constructor, a network is created with
            :python:`qnet_class(**qnet_kwargs, hparams=qnet_hparams)`.

        "replay_memory_type" : str or class or instance
            Replay memory class. Can be class, its name or module path,
            or a class instance.
            If class name is given, the class must be from module
            :mod:`texar.core` or :mod:`texar.custom`.
            Ignored if a `replay_memory` is given to the agent constructor.

        "replay_memory_hparams" : dict, optional
            Hyperparameters for the replay memory. With the
            :attr:`replay_memory_kwargs` argument to the constructor,
            a network is created with
            :python:`replay_memory_class(
            **replay_memory_kwargs, hparams=replay_memory_hparams)`.

        "exploration_type" : str or class or instance
            Exploration class. Can be class,
            its name or module path, or a class instance. If class name is
            given, the class must be from module :mod:`texar.core` or
            :mod:`texar.custom`. Ignored if a `exploration` is given to
            the agent constructor.

        "exploration_hparams" : dict, optional
            Hyperparameters for the exploration class.
            With the :attr:`exploration_kwargs` argument to the constructor,
            a network is created with :python:`exploration_class(
            **exploration_kwargs, hparams=exploration_hparams)`.

        "optimization" : dict
            Hyperparameters of optimization for updating the Q-net.
            See :func:`~texar.core.default_optimization_hparams` for details.

        "cold_start_steps": int
            In the beginning, Q-net is not trained in the first few steps.

        "sample_batch_size": int
            The number of samples taken in replay memory when training.

        "target_update_strategy": string

            - If **"copy"**, the target network is assigned with the parameter \
            of Q-net every :attr:`"update_period"` steps.

            - If **"tau"**, target will be updated by assigning as
            ``` (1 - 1/update_period) * target + 1/update_period * qnet ```

        "update_period": int
            Frequecy of updating the target network, i.e., updating
            the target once for every "update_period" steps.

        "discount_factor" : float
            The discount factor of reward.

        "name" : str
            Name of the agent.
        """
        return {
            'qnet_type': 'CategoricalQNet',
            'qnet_hparams': None,
            'replay_memory_type': 'DequeReplayMemory',
            'replay_memory_hparams': None,
            'exploration_type': 'EpsilonLinearDecayExploration',
            'exploration_hparams': None,
            'optimization': opt.default_optimization_hparams(),
            'target_update_strategy': 'copy',
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

            if self._target_update_strategy == 'copy':
                self._update_op = self._get_copy_update_op()
            elif self._target_update_strategy == 'tau':
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
            value_ = (1. - tau) * self._target.trainable_variables[i] + \
                    tau * self._qnet.trainable_variables[i]
            op.append(tf.assign(
                ref=self._target.trainable_variables[i], value=value_))
        return op

    def _observe(self, reward, terminal, train_policy, feed_dict):
        if self._timestep > self._cold_start_steps and train_policy:
            self._train_qnet(feed_dict)

        action_one_hot = [0.] * self._num_actions
        action_one_hot[self._action] = 1.

        self._replay_memory.add(dict(
            observ=self._observ,
            action=action_one_hot,
            reward=reward,
            terminal=terminal,
            next_observ=None))
        self._timestep += 1

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

        feed_dict_ = {
            self._observ_inputs: observ_batch,
            self._y_inputs: y_batch,
            self._action_inputs: action_batch
        }
        feed_dict_.update(feed_dict or {})

        self._sess.run(self._train_op, feed_dict=feed_dict_)

        self._update_target(feed_dict)

    def _update_target(self, feed_dict):
        if self._target_update_strategy == 'tau' or (
                self._target_update_strategy == 'copy' and
                self._timestep % self._update_period == 0):
            self._sess.run(self._update_op, feed_dict=feed_dict)

    def _qvalues_from_qnet(self, observ):
        return self._sess.run(
            self._qnet_outputs['qvalues'],
            feed_dict={self._observ_inputs: np.array([observ]),
                       tx.global_mode(): tf.estimator.ModeKeys.PREDICT})

    def _qvalues_from_target(self, observ):
        return self._sess.run(
            self._target_outputs['qvalues'],
            feed_dict={self._observ_inputs: np.array([observ]),
                       tx.global_mode(): tf.estimator.ModeKeys.PREDICT})

    def _update_observ_action(self, observ, action):
        self._observ = observ
        self._action = action
        if self._replay_memory.size() > 0:
            self._replay_memory.last()['next_observ'] = self._observ

    def _get_action(self, observ, feed_dict=None):
        qvalue = self._qvalues_from_qnet(observ)

        if random.random() < self._exploration.get_epsilon(self._timestep):
            action = random.randrange(self._num_actions)
        else:
            action = np.argmax(qvalue)

        self._update_observ_action(observ, action)

        return action

    def _reset(self):
        self._observ = None
        self._action = None

    @property
    def sess(self):
        """The tf session.
        """
        return self._sess

    @sess.setter
    def sess(self, session):
        self._sess = session
