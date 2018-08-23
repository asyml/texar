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
"""Policy Gradient agent for sequence prediction.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=too-many-instance-attributes, too-many-arguments, no-member

import tensorflow as tf

from texar.agents.seq_agent_base import SeqAgentBase
from texar.core import optimization as opt
from texar.losses.pg_losses import pg_loss_with_logits
from texar.losses.rewards import discount_reward
from texar.losses.entropy import sequence_entropy_with_logits

__all__ = [
    "SeqPGAgent"
]

class SeqPGAgent(SeqAgentBase):
    """Policy Gradient agent for sequence prediction.

    This is a wrapper of the **training process** that trains a model
    with policy gradient. Agent itself does not create new trainable variables.

    Args:
        samples: An `int` Tensor of shape `[batch_size, max_time]` containing
            sampled sequences from the model.
        logits: A float Tenosr of shape `[batch_size, max_time, vocab_size]`
            containing the logits of samples from the model.
        sequence_length: A Tensor of shape `[batch_size]`.
            Time steps beyond the respective sequence lengths are masked out.
        trainable_variables (optional): Trainable variables of the model to
            update during training. If `None`, all trainable variables in the
            graph are used.
        learning_rate (optional): Learning rate for policy optimization. If
            not given, determine the learning rate from :attr:`hparams`.
            See :func:`~texar.core.get_train_op` for more details.
        sess (optional): A tf session.
            Can be `None` here and set later with `agent.sess = session`.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    Example:

        .. code-block:: python

            ## Train a decoder with policy gradient
            decoder = BasicRNNDecoder(...)
            outputs, _, sequence_length = decoder(
                decoding_strategy='infer_sample', ...)

            sess = tf.Session()
            agent = SeqPGAgent(
                samples=outputs.sample_id,
                logits=outputs.logits,
                sequence_length=sequence_length,
                sess=sess)
            while training:
                # Generate samples
                vals = agent.get_samples()
                # Evaluate reward
                sample_text = tx.utils.map_ids_to_strs(vals['samples'], vocab)
                reward_bleu = []
                for y, y_ in zip(ground_truth, sample_text)
                    reward_bleu.append(tx.evals.sentence_bleu(y, y_)
                # Update
                agent.observe(reward=reward_bleu)
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
        pg_loss = pg_loss_with_logits(
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

        if self._hparams.entropy_weight > 0:
            entropy = self._get_entropy()
            pg_loss -= self._hparams.entropy_weight * entropy

        return pg_loss

    def _get_entropy(self):
        loss_hparams = self._hparams.loss
        return sequence_entropy_with_logits(
            self._logits,
            sequence_length=self._sequence_length,
            average_across_batch=loss_hparams.average_across_batch,
            average_across_timesteps=loss_hparams.average_across_timesteps,
            sum_over_batch=loss_hparams.sum_over_batch,
            sum_over_timesteps=loss_hparams.sum_over_timesteps,
            time_major=loss_hparams.time_major)

    def _get_train_op(self):
        train_op = opt.get_train_op(
            loss=self._pg_loss,
            variables=self._trainable_variables,
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
                'discount_factor': 0.95,
                'normalize_reward': False,
                'entropy_weight': 0.,
                'loss': {
                    'average_across_batch': True,
                    'average_across_timesteps': False,
                    'sum_over_batch': False,
                    'sum_over_timesteps': True,
                    'time_major': False
                },
                'optimization': default_optimization_hparams(),
                'name': 'pg_agent',
            }

        Here:

        "discount_factor" : float
            The discount factor of reward.

        "normalize_reward" : bool
            Whether to normalize the discounted reward, by
            `(discounted_reward - mean) / std`. Here `mean` and `std` are
            over all time steps and all samples in the batch.

        "entropy_weight" : float
            The weight of entropy loss of the sample distribution, to encourage
            maximizing the Shannon entropy. Set to 0 to disable the loss.

        "loss" : dict
            Extra keyword arguments for
            :func:`~texar.losses.pg_loss_with_logits`, including the
            reduce arguments (e.g., `average_across_batch`) and `time_major`

        "optimization" : dict
            Hyperparameters of optimization for updating the policy net.
            See :func:`~texar.core.default_optimization_hparams` for details.

        "name" : str
            Name of the agent.
        """
        return {
            'discount_factor': 0.95,
            'normalize_reward': False,
            'entropy_weight': 0.,
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
        """Returns sequence samples and extra results.

        Args:
            extra_fetches (dict, optional): Extra tensors to fetch values,
                besides `samples` and `sequence_length`. Same as the
                `fetches` argument of
                :tf_main:`tf.Session.run <Session#run>` and
                tf_main:`partial_run <Session#partial_run>`.
            feed_dict (dict, optional): A `dict` that maps tensor to
                values. Note that all placeholder values used in
                :meth:`get_samples` and subsequent :meth:`observe` calls
                should be fed here.

        Returns:
            A `dict` with keys **"samples"** and **"sequence_length"**
            containing the fetched values of :attr:`samples` and
            :attr:`sequence_length`, as well as other fetched values
            as specified in :attr:`extra_fetches`.

        Example:

            .. code-block:: python

                extra_fetches = {'truth_ids': data_batch['text_ids']}
                vals = agent.get_samples()
                sample_text = tx.utils.map_ids_to_strs(vals['samples'], vocab)
                truth_text = tx.utils.map_ids_to_strs(vals['truth_ids'], vocab)
                reward = reward_fn_in_python(truth_text, sample_text)
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

    def observe(self, reward, train_policy=True, compute_loss=True):
        """Observes the reward, and updates the policy or computes loss
        accordingly.

        Args:
            reward: A Python array/list of shape `[batch_size]` containing
                the reward for the samples generated in last call of
                :meth:`get_samples`.
            train_policy (bool): Whether to update the policy model according
                to the reward.
            compute_loss (bool): If `train_policy` is False, whether to
                compute the policy gradient loss (but does not update the
                policy).

        Returns:
            If `train_policy` or `compute_loss` is True, returns the loss
            (a python float scalar). Otherwise returns `None`.
        """
        self._rewards = reward

        if train_policy:
            return self._train_policy()
        elif compute_loss:
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
        """
        fetches = {
            "loss": self._train_op,
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
