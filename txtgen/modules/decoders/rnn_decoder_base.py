#
"""
Base class for RNN decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.seq2seq import Decoder as TFDecoder
from tensorflow.contrib.seq2seq import dynamic_decode

from txtgen.modules.module_base import ModuleBase
from txtgen.modules.decoders import rnn_decoder_helpers
from txtgen.core import layers, utils
from txtgen import context

# pylint: disable=not-context-manager, too-many-arguments

class RNNDecoderBase(ModuleBase, TFDecoder):
    """Base class inherited by all RNN decoder classes.

    Args:
        cell (RNNCell, optional): If not specified, a cell is created as
            specified in :attr:`hparams["rnn_cell"]`.
        embedding (optional): A `Variable` or a 2D `Tensor` (or `array`)
            of shape `[vocab_size, embedding_dim]` that contains the token
            embeddings. Ignore if :attr:`hparams["embedding_enabled"]`
            is `False`. If :attr:`hparams["embedding_enabled"]` is `True`:

            If a `Variable`, it is directly used in encoding, and
            hyperparameters in :attr:`hparams["embedding"]` are ignored.

            If a `Tensor` or `array`, it is used to initialize the token
            embedding variable. The :attr:`"initializer"` and :attr:`"dim"`
            hyperparameters in :attr:`hparams["embedding"]` are ignored.

            If not given, a new `Variable` is created as specified in
            :attr:`hparams["embedding"]`.
        vocab_size (int, optional): The vocabulary size. Required if
            :attr:`embedding` is not provided.
        hparams (dict, optional): Hyperparameters. If not specified, the default
            hyperparameter setting is used. See :attr:`default_hparams` for the
            structure and default values.
    """

    def __init__(self,
                 cell=None,
                 embedding=None,
                 vocab_size=None,
                 hparams=None):
        ModuleBase.__init__(self, hparams)

        self._helper = None
        self._initial_state = None

        # Make rnn cell
        with tf.variable_scope(self.variable_scope):
            if cell is not None:
                self._cell = cell
            else:
                self._cell = layers.get_rnn_cell(self._hparams.rnn_cell)

        # Make embedding
        if vocab_size is None:
            if self._hparams.embedding_enabled is False or embedding is None:
                raise ValueError(
                    "`vocab_size` is required if embedding is not enabled and ")

        self._embedding = None
        if self._hparams.embedding_enabled:
            if isinstance(embedding, tf.Variable):
                self._embedding = embedding
            else:
                self._embedding = layers.get_embedding(
                    self._hparams.embedding, embedding, vocab_size,
                    self.variable_scope)
            if self._hparams.embedding.trainable:
                self._add_trainable_variable(self._embedding)

        self._vocab_size = self._embedding.get_shape().as_list()[0]

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        The dictionary has the following structure and default values.

        See :meth:`~txtgen.core.layers.default_rnn_cell_hparams` for the
        default rnn cell hyperparameters, and
        :meth:`~txtgen.core.layers.default_embedding_hparams` for the default
        embedding hyperparameters.

        .. code-block:: python

            {
                # A dictionary of rnn cell hyperparameters. Ignored if `cell`
                # is given when constructing the encoder.
                "rnn_cell": txtgen.core.layers.default_rnn_cell_hparams(),

                # A dictionary of token embedding hyperparameters for embedding
                # initialization. Ignored if `embedding` is given and is
                # a tf.Variable when constructing the encoder. If `embedding`
                # is given and is a Tensor or numpy array, the "dim" and
                # "initializer" specs of embedding are ignored.
                "embedding": txtgen.core.layers.default_embedding_hparams(),

                # (optional) An integer. Maximum allowed number of decoding
                # steps at training time. If `None` (default), decoding is
                # performed until fully done, e.g., encountering the EOS token.
                "max_decoding_length_train": None,

                # (optional) An integer. Maximum allowed number of decoding
                # steps at inference time. If `None` (default), decoding is
                # performed until fully done, e.g., encountering the EOS token.
                "max_decoding_length_infer": None,

                # Name of the decoder.
                "name": "rnn_decoder"
            }
        """
        return {
            "rnn_cell": layers.default_rnn_cell_hparams(),
            "embedding": layers.default_embedding_hparams(),
            "helper_train": rnn_decoder_helpers.default_helper_train_hparams(),
            "helper_infer": rnn_decoder_helpers.default_helper_infer_hparams(),
            "max_decoding_length_train": None,
            "max_decoding_length_infer": None,
            "name": "rnn_decoder"
        }


    def _build(self, helper, initial_state):    # pylint: disable=W0221
        """Performs decoding.

        Args:
            helper: An instance of `tf.contrib.seq2seq.Helper` that helps with
                the decoding process. For example, use an instance of
                `TrainingHelper` in training phase.
            initial_state: Initial state of decoding.

        Returns:
            `(outputs, final_state, sequence_lengths)`: `outputs` is an object
            containing the decoder output on all time steps, `final_state` is
            the cell state of the final time step, `sequence_lengths` is a
            Tensor of shape `[batch_size]`.
        """
        self._helper = helper
        self._initial_state = initial_state

        max_decoding_length_train = self._hparams.max_decoding_length_train
        if max_decoding_length_train is None:
            max_decoding_length_train = utils.MAX_SEQ_LENGTH
        max_decoding_length_infer = self._hparams.max_decoding_length_infer
        if max_decoding_length_infer is None:
            max_decoding_length_infer = utils.MAX_SEQ_LENGTH
        max_decoding_length = tf.cond(
            context.is_train(),
            lambda: max_decoding_length_train,
            lambda: max_decoding_length_infer)
        outputs, final_state, sequence_lengths = dynamic_decode(
            decoder=self, maximum_iterations=max_decoding_length)

        self._add_internal_trainable_variables()
        # Add trainable variables of `self._cell` which may be constructed
        # externally
        self._add_trainable_variable(self._cell.trainable_variables)
        self._built = True

        return outputs, final_state, sequence_lengths


    @property
    def batch_size(self):
        return self._helper.batch_size

    def initialize(self, name=None):
        """Called before any decoding iterations.

        This methods must compute initial input values and initial state.

        Args:
            name (str, optional): Name scope for any created operations.

        Returns:
            `(finished, initial_inputs, initial_state)`: initial values of
            'finished' flags, inputs and state.
        """
        raise NotImplementedError

    def step(self, time, inputs, state, name=None):
        """Called per step of decoding (but only once for dynamic decoding).

        Args:
            time: Scalar `int32` tensor. Current step number.
            inputs: RNNCell input (possibly nested tuple of) tensor[s] for this
                time step.
            state: RNNCell state (possibly nested tuple of) tensor[s] from
                previous time step.
            name: Name scope for any created operations.

        Returns:
            `(outputs, next_state, next_inputs, finished)`: `outputs` is an
            object containing the decoder output, `next_state` is a (structure
            of) state tensors and TensorArrays, `next_inputs` is the tensor that
            should be used as input for the next step, `finished` is a boolean
            tensor telling whether the sequence is complete, for each sequence
            in the batch.
        """
        raise NotImplementedError

    def finalize(self, outputs, final_state, sequence_lengths):
        raise NotImplementedError

    @property
    def embedding(self):
        """The embedding variable.
        """
        return self._embedding

    @property
    def cell(self):
        """The RNN cell.
        """
        return self._cell

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
