#
"""Policy Gradient agent for sequence prediction.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=too-many-instance-attributes, too-many-arguments, no-member

import tensorflow as tf

from texar.agents.seq_agent_base import SeqAgentBase
from texar.core import optimization as opt
from texar.losses import pg_losses as losses
from texar.losses.rewards import discount_reward

__all__ = [
    "SeqPGAgent"
]

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

        self._lr = learning_rate

        # Tensors
        self._samples = samples
        self._logits = logits
        self._sequence_length = sequence_length
        self._trainable_variables = trainable_variables

        # Python values
        self._samples_py = None
        self._sequence_length_py = None
        self._rewards = None

        self._sess = sess

        # For session partial run
        self._partial_run_handle = None
        self._qvalue_inputs_fed = False

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
            'normalize_reward': False,
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

    def _get_partial_run_feeds(self, feeds=None):
        if feeds is None:
            feeds = []
        feeds += [self._qvalue_inputs]
        return feeds

    def _setup_partial_run(self, fetches=None, feeds=None):
        fetches_ = [self._samples, self._sequence_length, self._pg_loss,
                    self._train_op]
        if fetches is not None:
            for fet in fetches:
                if fet not in fetches_:
                    fetches_.append(fet)

        feeds = self._get_partial_run_feeds(feeds)

        self._partial_run_handle = self._sess.partial_run_setup(
            fetches_, feeds=feeds)

        self._qvalue_inputs_fed = False

    def _check_extra_fetches(self, extra_fetches):
        fetch_values = None
        if extra_fetches is not None:
            fetch_values = list(extra_fetches.values())
        if fetch_values is not None:
            if self._samples in fetch_values:
                raise ValueError(
                    "`samples` must not be included in `extra_fetches`. "
                    "It is added automatically.")
            if self._sequence_length in fetch_values:
                raise ValueError(
                    "`sequence_length` must not be included in `extra_fetches`."
                    " It is added automatically.")
            if "samples" in extra_fetches:
                raise ValueError(
                    "Key 'samples' is preserved and must not be used "
                    "in `extra_fetches`.")
            if "sequence_length" in extra_fetches:
                raise ValueError(
                    "Key 'sequence_length' is preserved and must not be used "
                    "in `extra_fetches`.")

    def get_samples(self, extra_fetches=None, feed_dict=None):
        """TODO
        """
        if self._sess is None:
            raise ValueError("`sess` must be specified before sampling.")

        self._check_extra_fetches(extra_fetches)

        # Sets up partial_run
        fetch_values = None
        if extra_fetches is not None:
            fetch_values = list(extra_fetches.values())
        feeds = None
        if feed_dict is not None:
            feeds = list(feed_dict.keys())
        self._setup_partial_run(fetches=fetch_values, feeds=feeds)

        # Runs the sampling
        fetches = {
            "samples": self._samples,
            "sequence_length": self._sequence_length
        }
        if extra_fetches is not None:
            fetches.update(extra_fetches)

        feed_dict_ = feed_dict

        vals = self._sess.partial_run(
            self._partial_run_handle, fetches, feed_dict=feed_dict_)

        self._samples_py = vals['samples']
        self._sequence_length_py = vals['sequence_length']

        return vals

    def observe(self, reward, train_policy=True, return_loss=True):
        """

        Args:
            reward: A Python array of shape `[batch_size]`.
            TODO
        """
        self._rewards = reward

        if train_policy:
            return self._train_policy()
        elif return_loss:
            return self._evaluate_pg_loss()
        else:
            return None

    def _get_qvalues(self):
        qvalues = discount_reward(
            self._rewards,
            self._sequence_length_py,
            discount=self._hparams.discount_factor,
            normalize=self._hparams.normalize_reward)
        return qvalues

    def _evaluate_pg_loss(self):
        fetches = {
            "loss": self._pg_loss
        }

        feed_dict_ = None
        if not self._qvalue_inputs_fed:
            qvalues = self._get_qvalues()
            feed_dict_ = {self._qvalue_inputs: qvalues}

        vals = self._sess.partial_run(
            self._partial_run_handle, fetches, feed_dict=feed_dict_)

        self._qvalue_inputs_fed = True

        return vals['loss']

    def _train_policy(self):
        """Updates the policy.

        Args:
            TODO
        """
        fetches = {
            "loss": self._train_op
        }

        feed_dict_ = None
        if not self._qvalue_inputs_fed:
            qvalues = self._get_qvalues()
            feed_dict_ = {self._qvalue_inputs: qvalues}

        vals = self._sess.partial_run(
            self._partial_run_handle, fetches, feed_dict=feed_dict_)

        self._qvalue_inputs_fed = True

        return vals['loss']

    @property
    def sess(self):
        """The tf session.
        """
        return self._sess

    @sess.setter
    def sess(self, sess):
        self._sess = sess

    @property
    def pg_loss(self):
        """The scalar tensor of policy gradient loss.
        """
        return self._pg_loss

    @property
    def sequence_length(self):
        """The tensor of sample sequence length, of shape `[batch_size]`.
        """
        return self._sequence_length

    @property
    def samples(self):
        """The tensor of sequence samples.
        """
        return self._samples

    @property
    def logits(self):
        """The tensor of sequence logits.
        """
        return self._logits
