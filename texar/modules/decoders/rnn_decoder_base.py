# Copyright 2019 The Texar Authors. All Rights Reserved.
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
Base class for RNN decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=too-many-arguments, no-name-in-module
# pylint: disable=too-many-branches, protected-access, too-many-locals
# pylint: disable=arguments-differ, unused-argument

import copy

import tensorflow as tf
from tensorflow.contrib.seq2seq import Decoder as TFDecoder
from tensorflow.contrib.seq2seq import dynamic_decode
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import nest

from texar.core import layers
from texar.utils import utils
from texar.utils.mode import is_train_mode, is_train_mode_py
from texar.module_base import ModuleBase
from texar.modules.decoders import rnn_decoder_helpers
from texar.utils.dtypes import is_callable
from texar.utils.shapes import shape_list
from texar.modules.decoders import tf_helpers as tx_helper

__all__ = [
    "RNNDecoderBase",
    "_make_output_layer"
]


def _make_output_layer_from_tensor(output_layer_tensor, vocab_size,
                                   output_layer_bias, variable_scope):
    """Creates a dense layer from a Tensor. Used to tie word embedding
    with the output layer weight.
    """
    affine_bias = None
    if output_layer_bias:
        with tf.variable_scope(variable_scope):
            affine_bias = tf.get_variable('affine_bias', [vocab_size])

    def _outputs_to_logits(outputs):
        shape = shape_list(outputs)
        dim = shape[-1]
        outputs = tf.reshape(outputs, [-1, dim])
        logits = tf.matmul(outputs, output_layer_tensor)
        if affine_bias is not None:
            logits += affine_bias
        logits = tf.reshape(logits, shape[:-1] + [vocab_size])
        return logits

    return _outputs_to_logits


def _make_output_layer(output_layer, vocab_size,
                       output_layer_bias, variable_scope):
    """Makes a decoder output layer.
    """
    _vocab_size = vocab_size
    if is_callable(output_layer):
        _output_layer = output_layer
    elif tf.contrib.framework.is_tensor(output_layer):
        _vocab_size = shape_list(output_layer)[1]
        _output_layer = _make_output_layer_from_tensor(
            output_layer, _vocab_size, output_layer_bias, variable_scope)
    elif output_layer is None:
        if _vocab_size is None:
            raise ValueError(
                "Either `output_layer` or `vocab_size` must be provided. "
                "Set `output_layer=tf.identity` if no output layer is "
                "wanted.")
        with tf.variable_scope(variable_scope):
            # pylint: disable=redefined-variable-type
            _output_layer = tf.layers.Dense(
                units=_vocab_size, use_bias=output_layer_bias)
    else:
        raise ValueError(
            "output_layer should be a callable layer, a tensor, or None. "
            "Unsupported type: ", type(output_layer)
        )

    return _output_layer, _vocab_size


class RNNDecoderBase(ModuleBase, TFDecoder):
    """Base class inherited by all RNN decoder classes.
    See :class:`~texar.modules.BasicRNNDecoder` for the argumenrts.

    See :meth:`_build` for the inputs and outputs of RNN decoders in general.

    .. document private functions
    .. automethod:: _build
    """

    def __init__(self,
                 cell=None,
                 vocab_size=None,
                 output_layer=None,
                 cell_dropout_mode=None,
                 hparams=None):
        ModuleBase.__init__(self, hparams)

        self._helper = None
        self._initial_state = None

        # Make rnn cell
        with tf.variable_scope(self.variable_scope):
            if cell is not None:
                self._cell = cell
            else:
                self._cell = layers.get_rnn_cell(
                    self._hparams.rnn_cell, cell_dropout_mode)
        self._beam_search_cell = None

        # Make the output layer
        self._output_layer, self._vocab_size = _make_output_layer(
            output_layer, vocab_size, self._hparams.output_layer_bias,
            self.variable_scope)

        self.max_decoding_length = None

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        The hyperparameters are the same as in
        :meth:`~texar.modules.BasicRNNDecoder.default_hparams` of
        :class:`~texar.modules.BasicRNNDecoder`, except that the default
        "name" here is "rnn_decoder".
        """
        return {
            "rnn_cell": layers.default_rnn_cell_hparams(),
            "helper_train": rnn_decoder_helpers.default_helper_train_hparams(),
            "helper_infer": rnn_decoder_helpers.default_helper_infer_hparams(),
            "max_decoding_length_train": None,
            "max_decoding_length_infer": None,
            "name": "rnn_decoder",
            "output_layer_bias": True,
        }

    def _build(self,
               decoding_strategy="train_greedy",
               initial_state=None,
               inputs=None,
               sequence_length=None,
               embedding=None,
               start_tokens=None,
               end_token=None,
               softmax_temperature=None,
               max_decoding_length=None,
               impute_finished=False,
               output_time_major=False,
               input_time_major=False,
               helper=None,
               mode=None,
               **kwargs):
        """Performs decoding. This is a shared interface for both
        :class:`~texar.modules.BasicRNNDecoder` and
        :class:`~texar.modules.AttentionRNNDecoder`.

        The function provides **3 ways** to specify the
        decoding method, with varying flexibility:

        1. The :attr:`decoding_strategy` argument: A string taking value of:

            - **"train_greedy"**: decoding in teacher-forcing fashion \
              (i.e., feeding \
              `ground truth` to decode the next step), and each sample is \
              obtained by taking the `argmax` of the RNN output logits. \
              Arguments :attr:`(inputs, sequence_length, input_time_major)` \
              are required for this strategy, and argument :attr:`embedding` \
              is optional.
            - **"infer_greedy"**: decoding in inference fashion (i.e., feeding \
              the `generated` sample to decode the next step), and each sample\
              is obtained by taking the `argmax` of the RNN output logits.\
              Arguments :attr:`(embedding, start_tokens, end_token)` are \
              required for this strategy, and argument \
              :attr:`max_decoding_length` is optional.
            - **"infer_sample"**: decoding in inference fashion, and each
              sample is obtained by `random sampling` from the RNN output
              distribution. Arguments \
              :attr:`(embedding, start_tokens, end_token)` are \
              required for this strategy, and argument \
              :attr:`max_decoding_length` is optional.

          This argument is used only when argument :attr:`helper` is `None`.

          Example:

            .. code-block:: python

                embedder = WordEmbedder(vocab_size=data.vocab.size)
                decoder = BasicRNNDecoder(vocab_size=data.vocab.size)

                # Teacher-forcing decoding
                outputs_1, _, _ = decoder(
                    decoding_strategy='train_greedy',
                    inputs=embedder(data_batch['text_ids']),
                    sequence_length=data_batch['length']-1)

                # Random sample decoding. Gets 100 sequence samples
                outputs_2, _, sequence_length = decoder(
                    decoding_strategy='infer_sample',
                    start_tokens=[data.vocab.bos_token_id]*100,
                    end_token=data.vocab.eos.token_id,
                    embedding=embedder,
                    max_decoding_length=60)

        2. The :attr:`helper` argument: An instance of subclass of \
           :class:`texar.modules.Helper`. This
           provides a superset of decoding strategies than above, for example:

            - :class:`~texar.modules.TrainingHelper` corresponding to the \
              "train_greedy" strategy.
            - :class:`~texar.modules.GreedyEmbeddingHelper` and \
              :class:`~texar.modules.SampleEmbeddingHelper` corresponding to \
              the "infer_greedy" and "infer_sample", respectively.
            - :class:`~texar.modules.TopKSampleEmbeddingHelper` for Top-K \
              sample decoding.
            - :class:`ScheduledEmbeddingTrainingHelper` and \
              :class:`ScheduledOutputTrainingHelper` for scheduled \
              sampling.
            - :class:`~texar.modules.SoftmaxEmbeddingHelper` and \
              :class:`~texar.modules.GumbelSoftmaxEmbeddingHelper` for \
              soft decoding and gradient backpropagation.

          Helpers give the maximal flexibility of configuring the decoding\
          strategy.

          Example:

            .. code-block:: python

                embedder = WordEmbedder(vocab_size=data.vocab.size)
                decoder = BasicRNNDecoder(vocab_size=data.vocab.size)

                # Teacher-forcing decoding, same as above with
                # `decoding_strategy='train_greedy'`
                helper_1 = texar.modules.TrainingHelper(
                    inputs=embedders(data_batch['text_ids']),
                    sequence_length=data_batch['length']-1)
                outputs_1, _, _ = decoder(helper=helper_1)

                # Gumbel-softmax decoding
                helper_2 = GumbelSoftmaxEmbeddingHelper(
                    embedding=embedder,
                    start_tokens=[data.vocab.bos_token_id]*100,
                    end_token=data.vocab.eos_token_id,
                    tau=0.1)
                outputs_2, _, sequence_length = decoder(
                    max_decoding_length=60, helper=helper_2)

        3. :attr:`hparams["helper_train"]` and :attr:`hparams["helper_infer"]`:\
           Specifying the helper through hyperparameters. Train and infer \
           strategy is toggled based on :attr:`mode`. Appriopriate arguments \
           (e.g., :attr:`inputs`, :attr:`start_tokens`, etc) are selected to \
           construct the helper. Additional arguments for helper constructor \
           can be provided either through :attr:`**kwargs`, or through \
           :attr:`hparams["helper_train/infer"]["kwargs"]`.

           This means is used only when both :attr:`decoding_strategy` and \
           :attr:`helper` are `None`.

           Example:

             .. code-block:: python

                h = {
                    "helper_infer": {
                        "type": "GumbelSoftmaxEmbeddingHelper",
                        "kwargs": { "tau": 0.1 }
                    }
                }
                embedder = WordEmbedder(vocab_size=data.vocab.size)
                decoder = BasicRNNDecoder(vocab_size=data.vocab.size, hparams=h)

                # Gumbel-softmax decoding
                output, _, _ = decoder(
                    decoding_strategy=None, # Sets to None explicit
                    embedding=embedder,
                    start_tokens=[data.vocab.bos_token_id]*100,
                    end_token=data.vocab.eos_token_id,
                    max_decoding_length=60,
                    mode=tf.estimator.ModeKeys.PREDICT)
                        # PREDICT mode also shuts down dropout

        Args:
            decoding_strategy (str): A string specifying the decoding
                strategy. Different arguments are required based on the
                strategy.
                Ignored if :attr:`helper` is given.
            initial_state (optional): Initial state of decoding.
                If `None` (default), zero state is used.

            inputs (optional): Input tensors for teacher forcing decoding.
                Used when `decoding_strategy` is set to "train_greedy", or
                when `hparams`-configured helper is used.

                - If :attr:`embedding` is `None`, `inputs` is directly \
                fed to the decoder. E.g., in `"train_greedy"` strategy, \
                `inputs` must be a 3D Tensor of shape \
                `[batch_size, max_time, emb_dim]` (or \
                `[max_time, batch_size, emb_dim]` if `input_time_major`==True).
                - If `embedding` is given, `inputs` is used as index \
                to look up embeddings and feed in the decoder. \
                E.g., if `embedding` is an instance of \
                :class:`~texar.modules.WordEmbedder`, \
                then :attr:`inputs` is usually a 2D int Tensor \
                `[batch_size, max_time]` (or \
                `[max_time, batch_size]` if `input_time_major`==True) \
                containing the token indexes.
            sequence_length (optional): A 1D int Tensor containing the
                sequence length of :attr:`inputs`.
                Used when `decoding_strategy="train_greedy"` or
                `hparams`-configured helper is used.
            embedding (optional): Embedding used when:

                - "infer_greedy" or "infer_sample" `decoding_strategy` is \
                used. This can be a callable or the `params` argument for \
                :tf_main:`embedding_lookup <nn/embedding_lookup>`. \
                If a callable, it can take a vector tensor of token `ids`, \
                or take two arguments (`ids`, `times`), where `ids` \
                is a vector tensor of token ids, and `times` is a vector tensor\
                of time steps (i.e., position ids). The latter case can be used\
                when attr:`embedding` is a combination of word embedding and\
                position embedding. `embedding` is required in this case.
                - "train_greedy" `decoding_strategy` is used.\
                This can be a callable or the `params` argument for \
                :tf_main:`embedding_lookup <nn/embedding_lookup>`. \
                If a callable, it can take :attr:`inputs` and returns \
                the input embedding. `embedding` is optional in this case.
            start_tokens (optional): A int Tensor of shape `[batch_size]`,
                the start tokens. Used when `decoding_strategy="infer_greedy"`
                or `"infer_sample"`, or when the helper specified in `hparams`
                is used.

                Example:

                    .. code-block:: python

                        data = tx.data.MonoTextData(hparams)
                        iterator = DataIterator(data)
                        batch = iterator.get_next()

                        bos_token_id = data.vocab.bos_token_id
                        start_tokens=tf.ones_like(batch['length'])*bos_token_id

            end_token (optional): A int 0D Tensor, the token that marks end
                of decoding.
                Used when `decoding_strategy="infer_greedy"` or
                `"infer_sample"`, or when the helper specified in `hparams`
                is used.
            softmax_temperature (optional): A float 0D Tensor, value to divide
                the logits by before computing the softmax. Larger values
                (above 1.0) result in more random samples. Must > 0. If `None`,
                1.0 is used.
                Used when `decoding_strategy="infer_sample"`.
            max_decoding_length: A int scalar Tensor indicating the maximum
                allowed number of decoding steps. If `None` (default), either
                `hparams["max_decoding_length_train"]` or
                `hparams["max_decoding_length_infer"]` is used
                according to :attr:`mode`.
            impute_finished (bool): If `True`, then states for batch
                entries which are marked as finished get copied through and
                the corresponding outputs get zeroed out.  This causes some
                slowdown at each time step, but ensures that the final state
                and outputs have the correct values and that backprop ignores
                time steps that were marked as finished.
            output_time_major (bool): If `True`, outputs are returned as
                time major tensors. If `False` (default), outputs are returned
                as batch major tensors.
            input_time_major (optional): Whether the :attr:`inputs` tensor is
                time major.
                Used when `decoding_strategy="train_greedy"` or
                `hparams`-configured helper is used.
            helper (optional): An instance of
                :class:`texar.modules.Helper`
                that defines the decoding strategy. If given,
                `decoding_strategy`
                and helper configs in :attr:`hparams` are ignored.
            mode (str, optional): A string taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`. If
                `TRAIN`, training related hyperparameters are used (e.g.,
                `hparams['max_decoding_length_train']`), otherwise,
                inference related hyperparameters are used (e.g.,
                `hparams['max_decoding_length_infer']`).
                If `None` (default), `TRAIN` mode is used.
            **kwargs: Other keyword arguments for constructing helpers
                defined by `hparams["helper_trainn"]` or
                `hparams["helper_infer"]`.

        Returns:
            `(outputs, final_state, sequence_lengths)`, where

            - **`outputs`**: an object containing the decoder output on all \
            time steps.
            - **`final_state`**: is the cell state of the final time step.
            - **`sequence_lengths`**: is an int Tensor of shape `[batch_size]` \
            containing the length of each sample.
        """
        # Helper
        if helper is not None:
            pass
        elif decoding_strategy is not None:
            if decoding_strategy == "train_greedy":
                helper = rnn_decoder_helpers._get_training_helper(
                    inputs, sequence_length, embedding, input_time_major)
            elif decoding_strategy == "infer_greedy":
                helper = tx_helper.GreedyEmbeddingHelper(
                    embedding, start_tokens, end_token)
            elif decoding_strategy == "infer_sample":
                helper = tx_helper.SampleEmbeddingHelper(
                    embedding, start_tokens, end_token, softmax_temperature)
            else:
                raise ValueError(
                    "Unknown decoding strategy: {}".format(decoding_strategy))
        else:
            if is_train_mode_py(mode):
                kwargs_ = copy.copy(self._hparams.helper_train.kwargs.todict())
                helper_type = self._hparams.helper_train.type
            else:
                kwargs_ = copy.copy(self._hparams.helper_infer.kwargs.todict())
                helper_type = self._hparams.helper_infer.type
            kwargs_.update({
                "inputs": inputs,
                "sequence_length": sequence_length,
                "time_major": input_time_major,
                "embedding": embedding,
                "start_tokens": start_tokens,
                "end_token": end_token,
                "softmax_temperature": softmax_temperature})
            kwargs_.update(kwargs)
            helper = rnn_decoder_helpers.get_helper(helper_type, **kwargs_)
        self._helper = helper

        # Initial state
        if initial_state is not None:
            self._initial_state = initial_state
        else:
            self._initial_state = self.zero_state(
                batch_size=self.batch_size, dtype=tf.float32)

        # Maximum decoding length
        max_l = max_decoding_length
        if max_l is None:
            max_l_train = self._hparams.max_decoding_length_train
            if max_l_train is None:
                max_l_train = utils.MAX_SEQ_LENGTH
            max_l_infer = self._hparams.max_decoding_length_infer
            if max_l_infer is None:
                max_l_infer = utils.MAX_SEQ_LENGTH
            max_l = tf.cond(is_train_mode(mode),
                            lambda: max_l_train, lambda: max_l_infer)
        self.max_decoding_length = max_l
        # Decode
        outputs, final_state, sequence_lengths = dynamic_decode(
            decoder=self, impute_finished=impute_finished,
            maximum_iterations=max_l, output_time_major=output_time_major)

        if not self._built:
            self._add_internal_trainable_variables()
            # Add trainable variables of `self._cell` which may be
            # constructed externally.
            self._add_trainable_variable(
                layers.get_rnn_cell_trainable_variables(self._cell))
            if isinstance(self._output_layer, tf.layers.Layer):
                self._add_trainable_variable(
                    self._output_layer.trainable_variables)
            # Add trainable variables of `self._beam_search_rnn_cell` which
            # may already be constructed and used.
            if self._beam_search_cell is not None:
                self._add_trainable_variable(
                    self._beam_search_cell.trainable_variables)

            self._built = True

        return outputs, final_state, sequence_lengths

    def _get_beam_search_cell(self, **kwargs):
        self._beam_search_cell = self._cell
        return self._cell

    def _rnn_output_size(self):
        size = self._cell.output_size
        if self._output_layer is tf.identity:
            return size
        else:
            # To use layer's compute_output_shape, we need to convert the
            # RNNCell's output_size entries into shapes with an unknown
            # batch size.  We then pass this through the layer's
            # compute_output_shape and read off all but the first (batch)
            # dimensions to get the output size of the rnn with the layer
            # applied to the top.
            output_shape_with_unknown_batch = nest.map_structure(
                lambda s: tensor_shape.TensorShape([None]).concatenate(s),
                size)
            layer_output_shape = self._output_layer.compute_output_shape(
                output_shape_with_unknown_batch)
            return nest.map_structure(lambda s: s[1:], layer_output_shape)

    @property
    def batch_size(self):
        return self._helper.batch_size

    @property
    def output_size(self):
        """Output size of one step.
        """
        raise NotImplementedError

    @property
    def output_dtype(self):
        """Types of output of one step.
        """
        raise NotImplementedError

    def initialize(self, name=None):
        # Inherits from TFDecoder
        # All RNN decoder classes must implement this
        raise NotImplementedError

    def step(self, time, inputs, state, name=None):
        # Inherits from TFDecoder
        # All RNN decoder classes must implement this
        raise NotImplementedError

    def finalize(self, outputs, final_state, sequence_lengths):
        # Inherits from TFDecoder
        # All RNN decoder classes must implement this
        raise NotImplementedError

    @property
    def cell(self):
        """The RNN cell.
        """
        return self._cell

    def zero_state(self, batch_size, dtype):
        """Zero state of the RNN cell.
        Equivalent to :attr:`decoder.cell.zero_state`.
        """
        return self._cell.zero_state(
            batch_size=batch_size, dtype=dtype)

    @property
    def state_size(self):
        """The state size of decoder cell.
        Equivalent to :attr:`decoder.cell.state_size`.
        """
        return self.cell.state_size

    @property
    def vocab_size(self):
        """The vocab size.
        """
        return self._vocab_size

    @property
    def output_layer(self):
        """The output layer.
        """
        return self._output_layer
