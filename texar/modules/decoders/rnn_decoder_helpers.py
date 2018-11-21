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
Various helper classes and utilities for RNN decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.seq2seq import TrainingHelper as TFTrainingHelper
from tensorflow.contrib.seq2seq import Helper as TFHelper
from tensorflow.contrib.distributions import RelaxedOneHotCategorical \
    as GumbelSoftmax

from texar.modules.embedders.embedder_base import EmbedderBase
from texar.utils import utils

# pylint: disable=not-context-manager, too-many-arguments
# pylint: disable=too-many-instance-attributes

__all__ = [
    "default_helper_train_hparams",
    "default_helper_infer_hparams",
    "get_helper",
    "_get_training_helper",
    "GumbelSoftmaxEmbeddingHelper",
    "SoftmaxEmbeddingHelper",
    "TeacherMaskSoftmaxEmbeddingHelper",
]

def default_helper_train_hparams():
    """Returns default hyperparameters of an RNN decoder helper in the training
    phase.

    See also :meth:`~texar.modules.decoders.rnn_decoder_helpers.get_helper`
    for information of the hyperparameters.

    Returns:
        dict: A dictionary with following structure and values:

        .. code-block:: python

            {
                # The `helper_type` argument for `get_helper`, i.e., the name
                # or full path to the helper class.
                "type": "TrainingHelper",

                # The `**kwargs` argument for `get_helper`, i.e., additional
                # keyword arguments for constructing the helper.
                "kwargs": {}
            }
    """
    return {
        "type": "TrainingHelper",
        "kwargs": {}
    }

def default_helper_infer_hparams():
    """Returns default hyperparameters of an RNN decoder helper in the inference
    phase.

    See also :meth:`~texar.modules.decoders.rnn_decoder_helpers.get_helper`
    for information of the hyperparameters.

    Returns:
        dict: A dictionary with following structure and values:

        .. code-block:: python

            {
                # The `helper_type` argument for `get_helper`, i.e., the name
                # or full path to the helper class.
                "type": "SampleEmbeddingHelper",

                # The `**kwargs` argument for `get_helper`, i.e., additional
                # keyword arguments for constructing the helper.
                "kwargs": {}
            }
    """
    return {
        "type": "SampleEmbeddingHelper",
        "kwargs": {}
    }


def get_helper(helper_type,
               inputs=None,
               sequence_length=None,
               embedding=None,
               start_tokens=None,
               end_token=None,
               **kwargs):
    """Creates a Helper instance.

    Args:
        helper_type: A :tf_main:`Helper <contrib/seq2seq/Helper>` class, its
            name or module path, or a class instance. If a class instance
            is given, it is returned directly.
        inputs (optional): Inputs to the RNN decoder, e.g., ground truth
            tokens for teacher forcing decoding.
        sequence_length (optional): A 1D int Tensor containing the
            sequence length of :attr:`inputs`.
        embedding (optional): A callable that takes a vector tensor of
            indexes (e.g., an instance of subclass of
            :class:`~texar.modules.EmbedderBase`), or the `params` argument
            for `embedding_lookup` (e.g., the embedding Tensor).
        start_tokens (optional): A int Tensor of shape `[batch_size]`,
            the start tokens.
        end_token (optional): A int 0D Tensor, the token that marks end
            of decoding.
        **kwargs: Additional keyword arguments for constructing the helper.

    Returns:
        A helper instance.
    """
    module_paths = [
        'texar.modules.decoders.rnn_decoder_helpers',
        'tensorflow.contrib.seq2seq',
        'texar.custom']
    class_kwargs = {"inputs": inputs,
                    "sequence_length": sequence_length,
                    "embedding": embedding,
                    "start_tokens": start_tokens,
                    "end_token": end_token}
    class_kwargs.update(kwargs)
    return utils.check_or_get_instance_with_redundant_kwargs(
        helper_type, class_kwargs, module_paths)


def _get_training_helper( #pylint: disable=invalid-name
        inputs, sequence_length, embedding=None, time_major=False, name=None):
    """Returns an instance of :tf_main:`TrainingHelper
    <contrib/seq2seq/TrainingHelper>` given embeddings.

    Args:
        inputs: If :attr:`embedding` is given, this is sequences of input
            token indexes. If :attr:`embedding` is `None`, this is passed to
            TrainingHelper directly.
        sequence_length (1D Tensor): Lengths of input token sequences.
        embedding (optional): The `params` argument of
            :tf_main:`tf.nn.embedding_lookup
            <nn/embedding_lookup>` (e.g., the embedding Tensor); or a callable
            that takes a vector of integer indexes and returns respective
            embedding (e.g., an instance of subclass of
            :class:`~texar.modules.EmbedderBase`).
        time_major (bool): Whether the tensors in `inputs` are time major.
            If `False` (default), they are assumed to be batch major.
        name (str, optional): Name scope for any created operations.

    Returns:
        An instance of TrainingHelper.

    Raises:
        ValueError: if `sequence_length` is not a 1D tensor.
    """
    if embedding is None:
        return TFTrainingHelper(inputs=inputs,
                                sequence_length=sequence_length,
                                time_major=time_major,
                                name=name)

    with tf.name_scope(name, "TrainingHelper", [embedding, inputs]):
        if callable(embedding):
            embedding_fn = embedding
        else:
            embedding_fn = (
                lambda ids: tf.nn.embedding_lookup(embedding, ids))
        emb_inputs = embedding_fn(inputs)
    helper = TFTrainingHelper(inputs=emb_inputs,
                              sequence_length=sequence_length,
                              time_major=time_major,
                              name=name)
    return helper


def get_embedding_and_fn(embedding):
    if isinstance(embedding, EmbedderBase):
        embedding = embedding.embedding

    if callable(embedding):
        raise ValueError("`embedding` must be an embedding tensor or an "
                         "instance of subclass of `EmbedderBase`.")
    else:
        return embedding, (lambda ids: tf.nn.embedding_lookup(embedding, ids))


class SoftmaxEmbeddingHelper(TFHelper):
    """A helper that feeds softmax probabilities over vocabulary
    to the next step.
    Uses the softmax probability vector to pass through word embeddings to
    get the next input (i.e., a mixed word embedding).

    A subclass of
    :tf_main:`Helper <contrib/seq2seq/Helper>`.
    Used as a helper to :class:`~texar.modules.RNNDecoderBase` :meth:`_build`
    in inference mode.

    Args:
        embedding: An embedding argument (:attr:`params`) for
            :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>`, or an
            instance of subclass of :class:`texar.modules.EmbedderBase`.
            Note that other callables are not acceptable here.
        start_tokens: An int tensor shaped `[batch_size]`. The
            start tokens.
        end_token: An int scalar tensor. The token that marks end of
            decoding.
        tau: A float scalar tensor, the softmax temperature.
        stop_gradient (bool): Whether to stop the gradient backpropagation
            when feeding softmax vector to the next step.
        use_finish (bool): Whether to stop decoding once `end_token` is
            generated. If `False`, decoding will continue until
            `max_decoding_length` of the decoder is reached.
    """

    def __init__(self, embedding, start_tokens, end_token, tau,
                 stop_gradient=False, use_finish=True):
        self._embedding, self._embedding_fn = get_embedding_and_fn(embedding)
        self._start_tokens = tf.convert_to_tensor(
            start_tokens, dtype=tf.int32, name="start_tokens")
        self._end_token = tf.convert_to_tensor(
            end_token, dtype=tf.int32, name="end_token")
        self._start_inputs = self._embedding_fn(self._start_tokens)
        self._batch_size = tf.size(self._start_tokens)
        self._tau = tau
        self._stop_gradient = stop_gradient
        self._use_finish = use_finish

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_dtype(self):
        return tf.float32

    @property
    def sample_ids_shape(self):
        return self._embedding.get_shape()[:1]

    def initialize(self, name=None):
        finished = tf.tile([False], [self._batch_size])
        return (finished, self._start_inputs)

    def sample(self, time, outputs, state, name=None):
        """Returns `sample_id` which is softmax distributions over vocabulary
        with temperature `tau`. Shape = `[batch_size, vocab_size]`
        """
        sample_ids = tf.nn.softmax(outputs / self._tau)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        if self._use_finish:
            hard_ids = tf.argmax(sample_ids, axis=-1, output_type=tf.int32)
            finished = tf.equal(hard_ids, self._end_token)
        else:
            finished = tf.tile([False], [self._batch_size])
        if self._stop_gradient:
            sample_ids = tf.stop_gradient(sample_ids)
        next_inputs = tf.matmul(sample_ids, self._embedding)
        return (finished, next_inputs, state)


class GumbelSoftmaxEmbeddingHelper(SoftmaxEmbeddingHelper):
    """A helper that feeds gumbel softmax sample to the next step.
    Uses the gumbel softmax vector to pass through word embeddings to
    get the next input (i.e., a mixed word embedding).

    A subclass of
    :tf_main:`Helper <contrib/seq2seq/Helper>`.
    Used as a helper to :class:`~texar.modules.RNNDecoderBase` :meth:`_build`
    in inference mode.

    Same as :class:`~texar.modules.SoftmaxEmbeddingHelper` except that here
    gumbel softmax (instead of softmax) is used.

    Args:
        embedding: An embedding argument (:attr:`params`) for
            :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>`, or an
            instance of subclass of :class:`texar.modules.EmbedderBase`.
            Note that other callables are not acceptable here.
        start_tokens: An int tensor shaped `[batch_size]`. The
            start tokens.
        end_token: An int scalar tensor. The token that marks end of
            decoding.
        tau: A float scalar tensor, the softmax temperature.
        straight_through (bool): Whether to use straight through gradient
            between time steps. If `True`, a single token with highest
            probability (i.e., greedy sample) is fed to the next step and
            gradient is computed using straight through. If `False` (default),
            the soft gumbel-softmax distribution is fed to the next step.
        stop_gradient (bool): Whether to stop the gradient backpropagation
            when feeding softmax vector to the next step.
        use_finish (bool): Whether to stop decoding once `end_token` is
            generated. If `False`, decoding will continue until
            `max_decoding_length` of the decoder is reached.
    """
    def __init__(self, embedding, start_tokens, end_token, tau,
                 straight_through=False, stop_gradient=False, use_finish=True):
        super(GumbelSoftmaxEmbeddingHelper, self).__init__(
            embedding, start_tokens, end_token, tau, stop_gradient, use_finish)
        self._straight_through = straight_through

    def sample(self, time, outputs, state, name=None):
        """Returns `sample_id` of shape `[batch_size, vocab_size]`. If
        `straight_through` is False, this is gumbel softmax distributions over
        vocabulary with temperature `tau`. If `straight_through` is True,
        this is one-hot vectors of the greedy samples.
        """
        sample_ids = tf.nn.softmax(outputs / self._tau)
        sample_ids = GumbelSoftmax(self._tau, logits=outputs).sample()
        if self._straight_through:
            size = tf.shape(sample_ids)[-1]
            sample_ids_hard = tf.cast(
                tf.one_hot(tf.argmax(sample_ids, -1), size), sample_ids.dtype)
            sample_ids = tf.stop_gradient(sample_ids_hard - sample_ids) \
                         + sample_ids
        return sample_ids


class TeacherMaskSoftmaxEmbeddingHelper(TFTrainingHelper):
    """A helper that implements the Teacher Mask described in the paper
    https://openreview.net/pdf?id=S1x2aiRqFX. In an unmasked step, it feeds
    softmax probabilities over vocabulary to the next step. In a masked step,
    it feeds the one-hot distribution of the target labels (:attr:`inputs`)
    to the next step.
    Uses the softmax probability or one-hot vector to pass through word
    embeddings to get the next input (i.e., a mixed word embedding).
    In this implementation, all sequences in a batch shares the same teacher
    mask.

    A subclass of
    :tf_main:`TrainingHelper <contrib/seq2seq/TrainingHelper>`.
    Used as a helper to :class:`~texar.modules.RNNDecoderBase` :meth:`_build`
    in training mode.

    Args:
        inputs (2D Tensor): Target sequence token indexes. It should be a tensor
            of shape `[batch_size, max_time]`. Must append both BOS and EOS
            tokens to each sequence.
        sequence_length (1D Tensor): Lengths of input token sequences. These
            lengths should include the BOS tokens but exclude the EOS tokens.
        embedding: An embedding argument (:attr:`params`) for
            :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>`, or an
            instance of subclass of :class:`texar.modules.EmbedderBase`.
            Note that other callables are not acceptable here.
        n_unmask: An int scalar tensor denotes the mask pattern together with
            :attr:`n_mask`. See the paper for details.
        n_mask: An int scalar tensor denotes the mask pattern together with
            :attr:`n_unmask`. See the paper for details.
        tau (float, optional): A float scalar tensor, the softmax temperature.
            Default to 1. 
        seed (int, optional): The random seed used to shift the mask.
        stop_gradient (bool): Whether to stop the gradient backpropagation
            when feeding softmax vector to the next step.
        name (str, optional): A name for the module.

    Example:

        .. code-block:: python

            embedder = WordEmbedder(vocab_size=data.vocab.size)
            decoder = BasicRNNDecoder(vocab_size=data.vocab.size)
            
            tm_helper = texar.modules.TeacherMaskSoftmaxEmbeddingHelper(
                inputs=data_batch['text_ids'],
                sequence_length=data_batch['length']-1,
                embedding=embedder,
                n_unmask=1,
                n_mask=0,
                tau=1.)

            outputs, _, _ = decoder(helper=tm_helper)

            loss = debleu(
                labels=data_batch['text_ids'][:, 1:],
                probs=outputs.sample_ids,
                sequence_length=data_batch['length']-1)

    """

    def __init__(self, inputs, sequence_length, embedding, n_unmask,
                 n_mask, tau=1., time_major=False, seed=None,
                 stop_gradient=False, name=None):
        with tf.variable_scope(name, "TeacherMaskSoftmaxEmbeddingHelper",
                               [embedding, tau, seed, stop_gradient]):
            super(TeacherMaskSoftmaxEmbeddingHelper, self).__init__(
                inputs=inputs,
                sequence_length=sequence_length,
                time_major=time_major)

            self._embedding, self._embedding_fn = get_embedding_and_fn(
                embedding)
            self._tau = tau
            self._seed = seed
            self._stop_gradient = stop_gradient

            self._zero_next_inputs = tf.zeros_like(
                self._embedding_fn(self._zero_inputs))

            self._n_unmask = n_unmask
            self._n_mask = n_mask
            self._n_cycle = tf.add(
                self._n_unmask, self._n_mask, name="n_cycle")
            self._n_shift = tf.random_uniform(
                [], maxval=self._n_cycle, dtype=self._n_cycle.dtype,
                seed=self._seed, name="n_shift")

    @property
    def sample_ids_dtype(self):
        return tf.float32

    @property
    def sample_ids_shape(self):
        return self._embedding.get_shape()[:1]

    @property
    def n_unmask(self):
        return self._n_unmask

    @property
    def n_mask(self):
        return self._n_mask

    def _is_masked(self, time):
        return (time + self._n_shift) % self._n_cycle < self._n_mask

    def initialize(self, name=None):
        finished = tf.equal(0, self._sequence_length)
        all_finished = tf.reduce_all(finished)
        next_inputs = tf.cond(
            all_finished,
            lambda: self._zero_next_inputs,
            lambda: self._embedding_fn(self._input_tas.read(0)))
        return (finished, next_inputs)

    def sample(self, time, outputs, state, name=None):
        """Returns `sample_id` of shape `[batch_size, vocab_size]`. In an
        unmasked step, it is softmax distributions over vocabulary with
        temperature :attr:`tau`; in a masked step, it is one-hot
        representations of :attr:`input` in the next step.
        """
        next_time = time + 1
        sample_ids = tf.cond(
            self._is_masked(next_time),
            lambda: tf.one_hot(self._input_tas.read(next_time),
                               self._embedding.get_shape()[0]),
            lambda: tf.nn.softmax(outputs / self._tau))
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        next_time = time + 1
        finished = (next_time >= self._sequence_length)
        all_finished = tf.reduce_all(finished)
        if self._stop_gradient:
            sample_ids = tf.stop_gradient(sample_ids)
        next_inputs = tf.cond(
            all_finished,
            lambda: self._zero_next_inputs,
            lambda: tf.matmul(sample_ids, self._embedding))
        return (finished, next_inputs, state)
