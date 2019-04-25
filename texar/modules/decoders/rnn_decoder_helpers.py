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
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.distributions import categorical
from tensorflow.contrib.distributions import RelaxedOneHotCategorical \
    as GumbelSoftmax

from texar.modules.decoders.tf_helpers import \
        Helper, TrainingHelper, GreedyEmbeddingHelper
from texar.modules.embedders.embedder_utils import soft_embedding_lookup
from texar.utils import utils

# pylint: disable=not-context-manager, too-many-arguments
# pylint: disable=too-many-instance-attributes

__all__ = [
    "default_helper_train_hparams",
    "default_helper_infer_hparams",
    "get_helper",
    "_get_training_helper",
    "TopKSampleEmbeddingHelper",
    "SoftmaxEmbeddingHelper",
    "GumbelSoftmaxEmbeddingHelper",
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
        'texar.modules.decoders.tf_helpers',
        #'tensorflow.contrib.seq2seq',
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
        return TrainingHelper(inputs=inputs,
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
    helper = TrainingHelper(inputs=emb_inputs,
                            sequence_length=sequence_length,
                            time_major=time_major,
                            name=name)
    return helper


def _top_k_logits(logits, k):
    """Adapted from
    https://github.com/openai/gpt-2/blob/master/src/sample.py#L63-L77
    """
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
        tf.equal(k, 0),
        lambda: logits,
        lambda: _top_k(),
    )


class TopKSampleEmbeddingHelper(GreedyEmbeddingHelper):
    """A helper for use during inference.

    Samples from `top_k` most likely candidates from a vocab distribution,
    and passes the result through an embedding layer to get the next input.
    """

    def __init__(self, embedding, start_tokens, end_token, top_k=10,
                 softmax_temperature=None, seed=None):
        """Initializer.

        Args:
            embedding: A callable or the `params` argument for
                `embedding_lookup`. If a callable, it can take a vector tensor
                of token `ids`, or take two arguments (`ids`, `times`),
                where `ids` is a vector
                tensor of token ids, and `times` is a vector tensor of current
                time steps (i.e., position ids). The latter case can be used
                when attr:`embedding` is a combination of word embedding and
                position embedding.
            start_tokens: `int32` vector shaped `[batch_size]`, the start
                tokens.
            end_token: `int32` scalar, the token that marks end of decoding.
            top_k: `int32` scalar tensor. Number of top candidates to sample
                from. Must be `>=0`. If set to 0, samples from all candidates
                (i.e., regular random sample decoding).
            softmax_temperature (optional): `float32` scalar, value to
                divide the logits by before computing the softmax. Larger values
                (above 1.0) result in more random samples, while smaller values
                push the sampling distribution towards the argmax. Must be
                strictly greater than 0. Defaults to 1.0.
            seed (optional): The sampling seed.

        Raises:
            ValueError: if `start_tokens` is not a 1D tensor or `end_token` is
            not a scalar.
        """
        super(TopKSampleEmbeddingHelper, self).__init__(
            embedding, start_tokens, end_token)
        self._top_k = top_k
        self._softmax_temperature = softmax_temperature
        self._seed = seed

    def sample(self, time, outputs, state, name=None):
        """Gets a sample for one step."""
        del time, state  # unused by sample_fn
        # Outputs are logits, we sample from the top_k candidates
        if not isinstance(outputs, tf.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))
        if self._softmax_temperature is None:
            logits = outputs
        else:
            logits = outputs / self._softmax_temperature

        logits = _top_k_logits(logits, k=self._top_k)

        sample_id_sampler = categorical.Categorical(logits=logits)
        sample_ids = sample_id_sampler.sample(seed=self._seed)

        return sample_ids


class SoftmaxEmbeddingHelper(Helper):
    """A helper that feeds softmax probabilities over vocabulary
    to the next step.
    Uses the softmax probability vector to pass through word embeddings to
    get the next input (i.e., a mixed word embedding).

    A subclass of
    :tf_main:`Helper <contrib/seq2seq/Helper>`.
    Used as a helper to :class:`~texar.modules.RNNDecoderBase` :meth:`_build`
    in inference mode.

    Args:
        embedding: A callable or the `params` argument for
            :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>`.
            If a callable, it can take a float tensor named `soft_ids` which is
            a distribution over indexes. For example, the shape of the tensor
            is typically `[batch_size, vocab_size]`. The callable can also
            take two arguments (`soft_ids`, `times`), where `soft_ids` is
            as above, and `times` is an int vector tensor of current
            time steps (i.e., position ids). The latter case can be used
            when attr:`embedding` is a combination of word embedding and
            position embedding.
        start_tokens: An int tensor shaped `[batch_size]`. The
            start tokens.
        end_token: An int scalar tensor. The token that marks end of
            decoding.
        tau: A float scalar tensor, the softmax temperature.
        embedding_size (optional): An int scalar tensor, the number of
            embedding vectors. Usually it is the vocab size. Required if
            :attr:`embedding` is a callable.
        stop_gradient (bool): Whether to stop the gradient backpropagation
            when feeding softmax vector to the next step.
        use_finish (bool): Whether to stop decoding once `end_token` is
            generated. If `False`, decoding will continue until
            `max_decoding_length` of the decoder is reached.
    """

    def __init__(self, embedding, start_tokens, end_token, tau,
                 embedding_size=None, stop_gradient=False, use_finish=True):
        if callable(embedding):
            self._embedding_fn = embedding

            if embedding_size is None:
                raise ValueError('`embedding_size` must be provided if '
                                 '`embedding` is a callable.')
            self._embedding_size = tf.convert_to_tensor(
                embedding_size, dtype=tf.int32, name="embedding_size")
        else:
            self._embedding_fn = (
                lambda soft_ids: soft_embedding_lookup(embedding, soft_ids))
            self._embedding_size = tf.shape(embedding)[0]

        self._start_tokens = tf.convert_to_tensor(
            start_tokens, dtype=tf.int32, name="start_tokens")
        self._end_token = tf.convert_to_tensor(
            end_token, dtype=tf.int32, name="end_token")
        if self._start_tokens.get_shape().ndims != 1:
            raise ValueError("start_tokens must be a vector")
        self._batch_size = array_ops.size(start_tokens)
        if self._end_token.get_shape().ndims != 0:
            raise ValueError("end_token must be a scalar")

        soft_start_tokens = tf.one_hot(
            self._start_tokens, self._embedding_size, dtype=tf.float32)
        self._embedding_args_cnt = len(utils.get_args(self._embedding_fn))
        if self._embedding_args_cnt == 1:
            self._start_inputs = self._embedding_fn(soft_ids=soft_start_tokens)
        elif self._embedding_args_cnt == 2:
            # Position index is 0 in the beginning
            times = tf.zeros([self._batch_size], dtype=tf.int32)
            self._start_inputs = self._embedding_fn(
                soft_ids=soft_start_tokens, times=times)
        else:
            raise ValueError('`embedding` should expect 1 or 2 arguments.')

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
        # A trick to convert a scalar Tensor `self._embedding_size` to
        # a `TensorShape`
        oh = tf.one_hot(0, self._embedding_size)
        return oh.get_shape()[:1]

    def initialize(self, name=None):
        finished = tf.tile([False], [self._batch_size])
        return (finished, self._start_inputs)

    def sample(self, time, outputs, state, name=None):
        """Returns `sample_id` which is softmax distributions over vocabulary
        with temperature `tau`. Shape = `[batch_size, vocab_size]`
        """
        sample_ids = tf.nn.softmax(outputs / self._tau)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None,
                    reach_max_time=None):
        if self._use_finish:
            hard_ids = tf.argmax(sample_ids, axis=-1, output_type=tf.int32)
            finished = tf.equal(hard_ids, self._end_token)
        else:
            finished = tf.tile([False], [self._batch_size])
        all_finished = tf.reduce_all(finished)

        if reach_max_time is not None:
            all_finished = tf.logical_or(all_finished, reach_max_time)

        if self._stop_gradient:
            sample_ids = tf.stop_gradient(sample_ids)

        if self._embedding_args_cnt == 1:
            del time, outputs  # unused by next_inputs_fn
            next_inputs = tf.cond(
                all_finished,
                # If we're finished, the next_inputs value doesn't matter
                lambda: self._start_inputs,
                lambda: self._embedding_fn(soft_ids=sample_ids))
        elif self._embedding_args_cnt == 2:
            # Prepare the position embedding of the next step
            times = tf.ones(self._batch_size, dtype=tf.int32) * (time+1)
            next_inputs = tf.cond(
                all_finished,
                # If we're finished, the next_inputs value doesn't matter
                lambda: self._start_inputs,
                lambda: self._embedding_fn(soft_ids=sample_ids, times=times))

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
        embedding: A callable or the `params` argument for
            :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>`.
            If a callable, it can take a float tensor named `soft_ids` which is
            a distribution over indexes. For example, the shape of the tensor
            is typically `[batch_size, vocab_size]`. The callable can also
            take two arguments (`soft_ids`, `times`), where `soft_ids` is
            as above, and `times` is an int vector tensor of current
            time steps (i.e., position ids). The latter case can be used
            when attr:`embedding` is a combination of word embedding and
            position embedding.
        start_tokens: An int tensor shaped `[batch_size]`. The
            start tokens.
        end_token: An int scalar tensor. The token that marks end of
            decoding.
        tau: A float scalar tensor, the softmax temperature.
        embedding_size (optional): An int scalar tensor, the number of
            embedding vectors. Usually it is the vocab size. Required if
            :attr:`embedding` is a callable.
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
                 embedding_size=None, straight_through=False,
                 stop_gradient=False, use_finish=True):
        super(GumbelSoftmaxEmbeddingHelper, self).__init__(
            embedding, start_tokens, end_token, tau, embedding_size,
            stop_gradient, use_finish)
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
