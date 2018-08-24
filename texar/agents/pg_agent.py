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
"""Policy Gradient agent.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=too-many-instance-attributes, too-many-arguments

import tensorflow as tf

from texar.agents.episodic_agent_base import EpisodicAgentBase
from texar.utils import utils
from texar.core import optimization as opt
from texar.losses import pg_losses as losses
from texar.losses.rewards import discount_reward


class PGAgent(EpisodicAgentBase):
    """Policy gradient agent for episodic setting. This agent here supports
    **un-batched** training, i.e., each time generates one action, takes one
    observation, and updates the policy.

    The policy must take in an observation of shape `[1] + observation_shape`,
    where the first dimension 1 stands for batch dimension, and output a `dict`
    containing:

    - Key **"action"** whose value is a Tensor of shape \
    `[1] + action_shape` containing a single action.
    - One of keys "log_prob" or "dist":

        - **"log_prob"**: A Tensor of shape `[1]`, the log probability of the \
        "action".
        - **"dist"**: A \
        tf_main:`tf.distributions.Distribution <distributions/Distribution>`\
        with the `log_prob` interface and \
        `log_prob = dist.log_prob(outputs["action"])`.

    .. role:: python(code)
       :language: python

    Args:
        env_config: An instance of :class:`~texar.agents.EnvConfig` specifying
            action space, observation space, and reward range, etc. Use
            :func:`~texar.agents.get_gym_env_config` to create an EnvConfig
            from a gym environment.
        sess (optional): A tf session.
            Can be `None` here and set later with `agent.sess = session`.
        policy (optional): A policy net that takes in observation and outputs
            actions and probabilities.
            If not given, a policy network is created based on :attr:`hparams`.
        policy_kwargs (dict, optional): Keyword arguments for policy
            constructor. Note that the `hparams` argument for network
            constructor is specified in the "policy_hparams" field of
            :attr:`hparams` and should not be included in `policy_kwargs`.
            Ignored if :attr:`policy` is given.
        policy_caller_kwargs (dict, optional): Keyword arguments for
            calling the policy to get actions. The policy is called with
            :python:`outputs=policy(inputs=observation, **policy_caller_kwargs)`
        learning_rate (optional): Learning rate for policy optimization. If
            not given, determine the learning rate from :attr:`hparams`.
            See :func:`~texar.core.get_train_op` for more details.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.
    """
    def __init__(self,
                 env_config,
                 sess=None,
                 policy=None,
                 policy_kwargs=None,
                 policy_caller_kwargs=None,
                 learning_rate=None,
                 hparams=None):
        EpisodicAgentBase.__init__(self, env_config, hparams)

        self._sess = sess
        self._lr = learning_rate
        self._discount_factor = self._hparams.discount_factor

        with tf.variable_scope(self.variable_scope):
            if policy is None:
                kwargs = utils.get_instance_kwargs(
                    policy_kwargs, self._hparams.policy_hparams)
                policy = utils.check_or_get_instance(
                    self._hparams.policy_type,
                    kwargs,
                    module_paths=['texar.modules', 'texar.custom'])
            self._policy = policy
            self._policy_caller_kwargs = policy_caller_kwargs or {}

        self._observs = []
        self._actions = []
        self._rewards = []

        self._train_outputs = None

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
            self._advantage_inputs = tf.placeholder(
                dtype=tf.float32,
                shape=[None, ],
                name='advantages_inputs')

            self._outputs = self._get_policy_outputs()

            self._pg_loss = self._get_pg_loss()

            self._train_op = self._get_train_op()

    def _get_policy_outputs(self):
        outputs = self._policy(
            inputs=self._observ_inputs, **self._policy_caller_kwargs)
        return outputs

    def _get_pg_loss(self):
        if 'log_prob' in self._outputs:
            log_probs = self._outputs['log_prob']
        elif 'dist' in self._outputs:
            log_probs = self._outputs['dist'].log_prob(self._action_inputs)
        else:
            raise ValueError('Outputs of the policy must have one of '
                             '"log_prob" or "dist".')
        pg_loss = losses.pg_loss_with_log_probs(
            log_probs=log_probs,
            advantages=self._advantage_inputs,
            average_across_timesteps=True,
            sum_over_timesteps=False)
        return pg_loss

    def _get_train_op(self):
        train_op = opt.get_train_op(
            loss=self._pg_loss,
            variables=self._policy.trainable_variables,
            learning_rate=self._lr,
            hparams=self._hparams.optimization.todict())
        return train_op

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values:

        .. role:: python(code)
           :language: python

        .. code-block:: python

            {
                'policy_type': 'CategoricalPolicyNet',
                'policy_hparams': None,
                'discount_factor': 0.95,
                'normalize_reward': False,
                'optimization': default_optimization_hparams(),
                'name': 'pg_agent',
            }

        Here:

        "policy_type" : str or class or instance
            Policy net. Can be class, its name or module path, or a class
            instance. If class name is given, the class must be from module
            :mod:`texar.modules` or :mod:`texar.custom`. Ignored if a
            `policy` is given to the agent constructor.

        "policy_hparams" : dict, optional
            Hyperparameters for the policy net. With the :attr:`policy_kwargs`
            argument to the constructor, a network is created with
            :python:`policy_class(**policy_kwargs, hparams=policy_hparams)`.

        "discount_factor" : float
            The discount factor of reward.

        "normalize_reward" : bool
            Whether to normalize the discounted reward, by
            `(discounted_reward - mean) / std`.

        "optimization" : dict
            Hyperparameters of optimization for updating the policy net.
            See :func:`~texar.core.default_optimization_hparams` for details.

        "name" : str
            Name of the agent.
        """
        return {
            'policy_type': 'CategoricalPolicyNet',
            'policy_hparams': None,
            'discount_factor': 0.95,
            'normalize_reward': False,
            'optimization': opt.default_optimization_hparams(),
            'name': 'pg_agent',
        }

    def _reset(self):
        self._observs = []
        self._actions = []
        self._rewards = []

    def _get_action(self, observ, feed_dict):
        fetches = {
            "action": self._outputs['action']
        }

        feed_dict_ = {self._observ_inputs: [observ, ]}
        feed_dict_.update(feed_dict or {})

        vals = self._sess.run(fetches, feed_dict=feed_dict_)
        action = vals['action']
        action = action[0] # Removes the batch dimension

        self._observs.append(observ)
        self._actions.append(action)

        return action

    def _observe(self, reward, terminal, train_policy, feed_dict):
        self._rewards.append(reward)

        if terminal and train_policy:
            self._train_policy(feed_dict=feed_dict)

    def _train_policy(self, feed_dict=None):
        """Updates the policy.

        Args:
            TODO
        """
        qvalues = discount_reward(
            [self._rewards], discount=self._hparams.discount_factor,
            normalize=self._hparams.normalize_reward)
        qvalues = qvalues[0, :]

        fetches = dict(loss=self._train_op)
        feed_dict_ = {
            self._observ_inputs: self._observs,
            self._action_inputs: self._actions,
            self._advantage_inputs: qvalues}
        feed_dict_.update(feed_dict or {})

        self._train_outputs = self._sess.run(fetches, feed_dict=feed_dict_)

    @property
    def sess(self):
        """The tf session.
        """
        return self._sess

    @sess.setter
    def sess(self, session):
        self._sess = session

    @property
    def policy(self):
        """The policy model.
        """
        return self._policy
