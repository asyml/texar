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
"""
Helper for interpolation algirithm.
New token is sample from model, ground_truth or reward according to lambdas
"""

import tensorflow as tf
import numpy as np

from tensorflow_probability import distributions as tfpd
from tensorflow.contrib.seq2seq import SampleEmbeddingHelper
from texar.tf.evals.bleu import sentence_bleu
from rouge import Rouge

rouge = Rouge()


def calc_reward(refs, hypo, unk_id, metric):
    """
    calculate the reward given hypo and refs and will return
    bleu score if metric is 'bleu' or return
    sum of (Rouge-1, Rouge-2, Rouge-L) if metric is 'rouge'
    """
    if len(hypo) == 0 or len(refs[0]) == 0:
        return 0.

    for i in range(len(hypo)):
        assert isinstance(hypo[i], int)
        if hypo[i] == unk_id:
            hypo[i] = -1

    if metric == 'bleu':
        return 0.01 * sentence_bleu(
            references=refs, hypothesis=hypo, smooth=True)
    else:
        ref_str = ' '.join([str(word) for word in refs[0]])
        hypo_str = ' '.join([str(word) for word in hypo])
        rouge_scores = \
            rouge.get_scores(hyps=[hypo_str], refs=[ref_str], avg=True)
        return sum([value['f'] for key, value in rouge_scores.items()])


class InterpolationHelper(SampleEmbeddingHelper):
    """
    Helper for interpolation algirithm.
    New token is sample from model, ground_truth or reward according to lambdas

    Args:
        embedding: A callable that takes a vector tensor of `ids` (argmax ids),
            or the `params` argument for `embedding_lookup`. The returned tensor
            will be passed to the decoder input.
        start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
        end_token: `int32` scalar, the token that marks end of decoding.
        vocab: texar.Vocab, the vocabularies of training set
        reward_metric: 'bleu' or 'rouge', the metric of reward
        ground_truth: the ground truth in training set
        ground_truth_length: the length of ground truth sentences
        lambdas: 'float32' vector shapes [3], according to which
            decide the way of generate the next token in training
    """
    def __init__(self,
                 embedding,
                 start_tokens,
                 end_token,
                 vocab,
                 reward_metric,
                 ground_truth,
                 ground_truth_length,
                 lambdas):
        SampleEmbeddingHelper.__init__(self, embedding, start_tokens, end_token)

        self._vocab = vocab
        self._ground_truth = ground_truth
        self._lambdas = lambdas
        self._ground_truth_length = ground_truth_length
        self._metric = reward_metric

    def sample(self, time, outputs, state, name=None):
        """
        sample tokens for next step, notice the special form
        of 'state'([decoded_ids, rnn_state])
        """
        sample_method_sampler = \
            tfpd.Categorical(probs=self._lambdas)
        sample_method_id = sample_method_sampler.sample()

        truth_feeding = lambda: tf.cond(
            tf.less(time, tf.shape(self._ground_truth)[1]),
            lambda: tf.cast(self._ground_truth[:, time], tf.int32),
            lambda: tf.ones_like(self._ground_truth[:, 0],
                                 dtype=tf.int32) * self._vocab.eos_token_id)

        self_feeding = lambda: SampleEmbeddingHelper.sample(
            self, time, outputs, state, name)

        reward_feeding = lambda: self._sample_by_reward(time, state)

        sample_ids = tf.cond(
            tf.logical_or(tf.equal(time, 0), tf.equal(sample_method_id, 1)),
            truth_feeding,
            lambda: tf.cond(
                tf.equal(sample_method_id, 2),
                reward_feeding,
                self_feeding))
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """
        notice the special form of 'state'([decoded_ids, rnn_state])
        """
        finished, next_inputs, next_state = SampleEmbeddingHelper.next_inputs(
            self, time, outputs, state[1], sample_ids, name)

        next_state = [tf.concat(
            [state[0][:, :time], tf.expand_dims(sample_ids, 1),
             state[0][:, time + 1:]], axis=1), next_state]
        next_state[0] = tf.reshape(next_state[0], (tf.shape(sample_ids)[0], 60))

        return finished, next_inputs, next_state

    def _sample_by_reward(self, time, state):
        def _get_rewards(time, prefix_ids, target_ids, ground_truth_length):
            batch_size = np.shape(target_ids)[0]
            words_in_target = \
                [np.unique(target_ids[i]) for i in range(batch_size)]
            unk_id = self._vocab.unk_token_id
            eos_id = self._vocab.eos_token_id

            # before append
            baseline_scores = []
            baseline_ids = prefix_ids[:, :time]
            for i in range(batch_size):
                ref = target_ids[i].tolist()
                if self._vocab.eos_token_id in ref:
                    ref = ref[:ref.index(self._vocab.eos_token_id)]

                hypo = baseline_ids[i].tolist()
                if self._vocab.eos_token_id in hypo:
                    hypo = hypo[:hypo.index(self._vocab.eos_token_id)]

                baseline_scores.append(calc_reward(
                    refs=[ref], hypo=hypo, unk_id=unk_id,
                    metric=self._metric))

            # append UNK
            syn_ids = np.concatenate([
                prefix_ids[:, :time],
                np.ones((batch_size, 1), dtype=np.int32) * unk_id], axis=1)

            reward_unk = []
            for i in range(batch_size):
                ref = target_ids[i].tolist()
                if self._vocab.eos_token_id in ref:
                    ref = ref[:ref.index(self._vocab.eos_token_id)]

                hypo = syn_ids[i].tolist()
                if self._vocab.eos_token_id in hypo:
                    hypo = hypo[:hypo.index(self._vocab.eos_token_id)]

                reward = calc_reward(refs=[ref], hypo=hypo, unk_id=unk_id,
                                     metric=self._metric)
                reward_unk.append(
                    np.ones((1, self._vocab.size), dtype=np.float32) *
                    reward - baseline_scores[i])
            result = np.concatenate(reward_unk, axis=0)

            # append tokens
            for i in range(batch_size):
                for id in words_in_target[i]:
                    if id == unk_id:
                        continue

                    syn_id = np.concatenate(
                        [prefix_ids[i:i + 1, :time], np.array([[id, ]])],
                        axis=1)
                    hypo = syn_id[0].tolist()
                    if self._vocab.eos_token_id in hypo:
                        hypo = hypo[:hypo.index(self._vocab.eos_token_id)]

                    ref = target_ids[i].tolist()
                    if self._vocab.eos_token_id in ref:
                        ref = ref[:ref.index(self._vocab.eos_token_id)]

                    dup = 1. if prefix_ids[i][time] == id and \
                                id != unk_id else 0.
                    eos = 1. if time < ground_truth_length[i] - 1 and \
                                id == eos_id else 0.

                    reward = calc_reward(
                        refs=[ref], hypo=hypo, unk_id=unk_id,
                        metric=self._metric)
                    result[i][id] = reward - baseline_scores[i] - dup - eos

            return result

        sampler = tfpd.Categorical(
            logits=tf.py_func(_get_rewards, [
                time, state[0], self._ground_truth,
                self._ground_truth_length], tf.float32))
        return tf.reshape(
            sampler.sample(), (tf.shape(self._ground_truth)[0],))
