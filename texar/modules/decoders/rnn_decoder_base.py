#
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

__all__ = [
    "RNNDecoderBase"
]

class RNNDecoderBase(ModuleBase, TFDecoder):
    """Base class inherited by all RNN decoder classes.

    See :class:`~texar.modules.BasicRNNDecoder` for the argumenrts.
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
        self._vocab_size = vocab_size
        self._output_layer = output_layer
        if output_layer is None:
            if self._vocab_size is None:
                raise ValueError(
                    "Either `output_layer` or `vocab_size` must be provided. "
                    "Set `output_layer=tf.identity` if no output layer is "
                    "wanted.")
            with tf.variable_scope(self.variable_scope):
                self._output_layer = tf.layers.Dense(units=self._vocab_size)
        elif output_layer is not tf.identity:
            if not isinstance(output_layer, tf.layers.Layer):
                raise ValueError(
                    "`output_layer` must be either `tf.identity` or "
                    "an instance of `tf.layers.Layer`.")

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        The hyperparameters have the same structure as in
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
            "name": "rnn_decoder"
        }

    def _build(self,
               initial_state=None,
               max_decoding_length=None,
               impute_finished=False,
               output_time_major=False,
               decoding_strategy="train_greedy",
               inputs=None,
               sequence_length=None,
               input_time_major=False,
               embedding=None,
               start_tokens=None,
               end_token=None,
               softmax_temperature=None,
               helper=None,
               mode=None,
               **kwargs):
        """Performs decoding. The decoder provides 3 ways to specify the
        decoding strategy, with varying flexibility:

        - :attr:`decoding_strategy` argument: A string taking value:

            - "train_greedy": decoding in training fashion (i.e., feeding \
              ground truth to decode the next step), and each sample is \
              obtained by taking the argmax of the RNN output logits. \
              Arguments :attr:`(inputs, sequence_length, input_time_major)` \
              are required for this strategy, and argument :attr:`embedding` \
              is optional.
            - "infer_greedy": decoding in inference fashion (i.e., feeding \
              the generated sample to decode the next step), and each sample\
              is obtained by taking the argmax of the RNN output logits.\
              Arguments :attr:`(embedding, start_tokens, end_token)` are \
              required for this strategy.
            - "infer_sample": decoding in inference fashion, and each sample \
              is obtained by random sampling from the RNN output distribution. \
              Arguments :attr:`(embedding, start_tokens, end_token)` are \
              required for this strategy.

          This argument is used only when :attr:`helper` is `None`.

        - :attr:`helper` argument: An instance of \
          :tf_main:`tf.contrib.seq2seq.Helper <contrib/seq2seq/Helper>`. This \
          provides a superset of decoding strategies than above, for example:

            - :tf_main:`TrainingHelper
              <contrib/seq2seq/TrainingHelper>` corresponding to the \
              :attr:`"train_argmax"` strategy.
            - :tf_main:`ScheduledEmbeddingTrainingHelper
              <contrib/seq2seq/ScheduledEmbeddingTrainingHelper>` and \
              :tf_main:`ScheduledOutputTrainingHelper
              <contrib/seq2seq/ScheduledOutputTrainingHelper>` for scheduled \
              sampling.
            - :class:`~texar.modules.SoftmaxEmbeddingHelper` and \
              :class:`~texar.modules.GumbelSoftmaxEmbeddingHelper` for \
              soft decoding and gradient backpropagation.

          This means gives the maximal flexibility of configuring the decoding\
          strategy.

        - :attr:`hparams["helper_train"]` and :attr:`hparams["helper_infer"]`:\
          Specifying the helper through hyperparameters. Train and infer \
          strategy is toggled based on :attr:`mode`. Appriopriate arguments \
          (e.g., :attr:`inputs`, :attr:`start_tokens`, etc) are selected to \
          construct the helper. Additional construction arguments can be \
          provided either through :attr:`**kwargs`, or through \
          :attr:`hparams["helper_train/infer"]["kwargs"]`.

          This means is used only when :attr:`decoding_strategy` and \
          :attr:`helper` are both `None`.

        Args:
            initial_state (optional): Initial state of decoding.
                If `None` (default), zero state is used.
            max_decoding_length: A int scalar Tensor indicating the maximum
                allowed number of decoding steps. If `None` (default), either
                :attr:`hparams["max_decoding_length_train"]` or
                :attr:`hparams["max_decoding_length_infer"]` is used
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
            decoding_strategy (str, optional): A string specifying the decoding
                strategy. Different arguments are required based on the
                strategy.
                Ignored if :attr:`helper` is given.
            inputs (optional): Input tensors. Used when
                :attr:`decoding_strategy="train_greedy"` or
                :attr:`hparams`-configured helper is used.

                If :attr:`embedding` is `None`, :attr:`inputs` is directly
                fed to the decoder. E.g., in `"train_greedy"` strategy,
                :attr:`inputs` must be a 3D Tensor of shape
                `[batch_size, max_time, emb_dim]` (or
                `[max_time, batch_size, emb_dim]` if :attr:`input_time_major`
                is `True`).

                If :attr:`embedding` is given, :attr:`inputs` is used as index
                to look up embeddings to be fed in the decoder. Requirements on
                :attr:`inputs` depend on :attr:`embedding`.
                E.g., if :attr:`embedding` is an instance of
                :class:`~texar.modules.WordEmbedder`,
                then :attr:`inputs` is usually a 2D int Tensor
                `[batch_size, max_time]` (or
                `[max_time, batch_size]` if :attr:`input_time_major`
                is `True`) containing the token indexes.
            sequence_length (optional): A 1D int Tensor containing the
                sequence length of :attr:`inputs`.
                Used when :attr:`decoding_strategy="train_greedy"` or
                :attr:`hparams`-configured helper is used.
            input_time_major (optional): Whether the :attr:`inputs` tensor is
                time major.
                Used when :attr:`decoding_strategy="train_greedy"` or
                :attr:`hparams`-configured helper is used.
            embedding (optional): A callable that returns embedding vectors
                of inputs, or the :attr:`params` argument of
                :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>`. In the
                later case, :attr:`inputs` (if used) must be a int Tensor
                containing the ids to be looked up in :attr:`embedding`.
                Required when :attr:`decoding_strategy="infer_greedy"`
                or `"infer_sample"`; optional when
                :attr:`decoding_strategy="train_greedy"`.
            start_tokens (optional): A int Tensor of shape `[batch_size]`,
                the start tokens.
                Used when :attr:`decoding_strategy="infer_greedy"` or
                `"infer_sample"`, or when :attr:`hparams`-configured
                helper is used.
            end_token (optional): A int 0D Tensor, the token that marks end
                of decoding.
                Used when :attr:`decoding_strategy="infer_greedy"` or
                `"infer_sample"`, or when :attr:`hparams`-configured
                helper is used.
            softmax_temperature (optional): A float 0D Tensor, value to divide
                the logits by before computing the softmax. Larger values
                (above 1.0) result in more random samples. Must > 0. If `None`,
                1.0 is used. Used when :attr:`decoding_strategy="infer_sample"`.
            helper (optional): An instance of
                :tf_main:`Helper <contrib/seq2seq/Helper>`
                that defines the decoding strategy. If given,
                :attr:`decoding_strategy`
                and helper configs in :attr:`hparams` are ignored.
            mode (str, optional): A string taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`. If
                `TRAIN`, training related hyperparameters are used (e.g.,
                :attr:`hparams['max_decoding_length_train']`), otherwise,
                inference related hyperparameters are used (e.g.,
                :attr:`hparams['max_decoding_length_infer']`). If
                `None` (default), `TRAIN` mode is used.
            **kwargs: Other keyword arguments for constructing helper
                defined by :attr:`hparams["helper_trainn"]` or
                :attr:`hparams["helper_infer"]`.

        Returns:
            `(outputs, final_state, sequence_lengths)`: `outputs` is an object
            containing the decoder output on all time steps, `final_state` is
            the cell state of the final time step, `sequence_lengths` is a
            Tensor of shape `[batch_size]`.
        """
        # Helper
        if helper is not None:
            pass
        elif decoding_strategy is not None:
            if decoding_strategy == "train_greedy":
                helper = rnn_decoder_helpers._get_training_helper(
                    inputs, sequence_length, embedding, input_time_major)
            elif decoding_strategy == "infer_greedy":
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding, start_tokens, end_token)
            elif decoding_strategy == "infer_sample":
                helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
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
        """Zero state of the rnn cell.

        Same as :attr:`decoder.cell.zero_state`.
        """
        return self._cell.zero_state(
            batch_size=batch_size, dtype=dtype)

    @property
    def state_size(self):
        """The state size of decoder cell.

        Same as :attr:`decoder.cell.state_size`.
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
