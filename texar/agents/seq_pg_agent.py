#
"""Policy Gradient agent for sequence prediction.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=too-many-instance-attributes, too-many-arguments, no-member

import numpy as np

import tensorflow as tf

from texar.agents.seq_agent_base import SeqAgentBase
from texar.core import optimization as opt
from texar.losses import pg_losses as losses

class SeqPGAgent(SeqAgentBase):
    """Policy Gradient agent for sequence prediction.

    Args:
        TODO
    """
    def __init__(self,
                 samples,
                 logits,
                 sequence_length,
                 trainable_variables=None,
                 learning_rate=None,
                 sess=None,
                 hparams=None):
        SeqAgentBase.__init__(self, hparams)

        self._sess = sess
        self._lr = learning_rate

        self._samples = samples
        self._logits = logits
        self._sequence_length = sequence_length
        self._trainable_variables = trainable_variables

        self._samples_py = None
        self._rewards = None

        self._build_graph()

    def _build_graph(self):
        with tf.variable_scope(self.variable_scope):
            self._qvalue_inputs = tf.placeholder(
                dtype=tf.float32,
                shape=[None, None],
                name='qvalue_inputs')

            self._pg_loss = self._get_pg_loss()

            self._train_op = self._get_train_op()

    def _get_pg_loss(self):
        loss_hparams = self._hparams.loss
        pg_loss = losses.pg_loss_with_logits(
            actions=self._samples,
            logits=self._logits,
            sequence_length=self._sequence_length,
            advantages=self._qvalue_inputs,
            batched=True,
            average_across_batch=loss_hparams.average_across_batch,
            average_across_timesteps=loss_hparams.average_across_timesteps,
            sum_over_batch=loss_hparams.sum_over_batch,
            sum_over_timesteps=loss_hparams.sum_over_timesteps,
            time_major=loss_hparams.time_major)
        return pg_loss

    def _get_train_op(self):
        train_op = opt.get_train_op(
            loss=self._pg_loss,
            variables=self._trainable_variables,
            learning_rate=self._lr,
            hparams=self._hparams.optimization.todict())
        return train_op

    @staticmethod
    def default_hparams():
        return {
            'discount_factor': 0.95,
            'policy_type': 'CategoricalPolicyNet',
            'policy_hparams': None,
            'loss': {
                'average_across_batch': True,
                'average_across_timesteps': False,
                'sum_over_batch': False,
                'sum_over_timesteps': True,
                'time_major': False
            },
            'optimization': opt.default_optimization_hparams(),
            'name': 'pg_agent',
        }

    def get_samples(self, feed_dict):
        """TODO
        """
        if self._sess is None:
            raise ValueError('`sess` must be specified before sampling.')

        fetches = dict(samples=self._samples)
        feed_dict_ = feed_dict

        vals = self._sess.run(fetches, feed_dict=feed_dict_)
        samples = vals['samples']

        self._samples_py = samples

        return samples

    #TODO(zhiting): Allow local rewards
    def observe(self, reward, train_policy, feed_dict):
        """

        Args:
            reward: A Python array of shape `[batch_size]`.
            TODO
        """
        self._rewards = reward

        if train_policy:
            self._train_policy(feed_dict=feed_dict)
        else:
            pass
            #TODO(zhiting): return policy loss

    def _train_policy(self, feed_dict=None):
        """Updates the policy.

        Args:
            TODO
        """
        discount_factor = self._hparams.discount_factor

        qvalues = np.array(self._rewards)
        qvalues = np.expand_dims(qvalues, -1)
        max_seq_length = self._samples_py.shape[1]
        if max_seq_length > 1:
            prefix = np.zeros(
                [qvalues.shape[0], max_seq_length-1], dtype=qvalues.dtype)
            qvalues = np.concatenate([prefix, qvalues])
            for i in range(max_seq_length - 2, -1, -1):
                qvalues[:, i] += discount_factor * qvalues[:, i + 1]

        q_mean = np.mean(qvalues)
        q_std = np.std(qvalues)
        qvalues = [(q - q_mean) / q_std for q in qvalues]

        fetches = dict(loss=self._train_op)
        feed_dict_ = {
            self._qvalue_inputs: qvalues
        }
        feed_dict_.update(feed_dict or {})

        vals = self._sess.run(fetches, feed_dict=feed_dict_)

        return vals['loss']

    @property
    def sess(self):
        """The tf session.
        """
        return self._sess

    @sess.setter
    def sess(self, session):
        self._sess = session
