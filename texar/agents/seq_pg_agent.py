#
"""Policy Gradient agent for sequence prediction.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=too-many-instance-attributes, too-many-arguments

import numpy as np

import tensorflow as tf

from texar.agents.agent_base import AgentBase
from texar.utils import utils
from texar.core import optimization as opt
from texar.losses import pg_losses as losses

# TODO(zhiting): Inherits from PGAgent ?
class SeqPGAgent(AgentBase):
    """Policy Gradient agent for sequence prediction.

    Args:
        TODO
    """
    def __init__(self,
                 sess=None,
                 policy=None,
                 policy_kwargs=None,
                 policy_caller_kwargs=None,
                 learning_rate=None,
                 hparams=None):
        AgentBase.__init__(self, None, hparams)

        self._sess = sess
        self._lr = learning_rate

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
            self._qvalue_inputs = tf.placeholder(
                dtype=tf.float32,
                shape=[None, ],
                name='qvalue_inputs')

            self._outputs = self._get_policy_outputs()

            self._pg_loss = self._get_pg_loss()

            self._train_op = self._get_train_op()

    def _get_policy_outputs(self):
        outputs = self._policy(
            inputs=self._observ_inputs, **self._policy_caller_kwargs)
        return outputs

    def _get_pg_loss(self):
        log_probs = self._outputs['dist'].log_prob(self._action_inputs)
        pg_loss = losses.pg_loss_with_log_probs(
            log_probs=log_probs, advantages=self._qvalue_inputs)
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
        return {
            'discount_factor': 0.95,
            'policy_type': 'CategoricalPolicyNet',
            'policy_hparams': None,
            'optimization': opt.default_optimization_hparams(),
            'name': 'pg_agent',
        }

    def _reset(self):
        self._observs = []
        self._actions = []
        self._rewards = []

    def _get_action(self, observ, feed_dict):
        fetches = dict(action=self._outputs['action'])

        feed_dict_ = {self._observ_inputs: [observ, ]}
        feed_dict_.update(feed_dict or {})

        vals = self._sess.run(fetches, feed_dict=feed_dict_)
        action = vals['action']

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
        discount_factor = self._hparams.discount_factor
        qvalues = list(self._rewards)
        for i in range(len(qvalues) - 2, -1, -1):
            qvalues[i] += discount_factor * qvalues[i + 1]

        q_mean = np.mean(qvalues)
        q_std = np.std(qvalues)
        qvalues = [(q - q_mean) / q_std for q in qvalues]

        fetches = dict(loss=self._train_op)
        feed_dict_ = {
            self._observ_inputs: self._observs,
            self._action_inputs: self._actions,
            self._qvalue_inputs: qvalues
        }
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
