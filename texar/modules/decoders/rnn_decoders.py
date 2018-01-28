#
"""
Various RNN decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=no-name-in-module, too-many-arguments, too-many-locals
# pylint: disable=not-context-manager, protected-access

import collections

import tensorflow as tf
from tensorflow.contrib.seq2seq import AttentionWrapper
from tensorflow.python.util import nest

from texar.modules.decoders.rnn_decoder_base import RNNDecoderBase
from texar.core.utils import get_instance, get_class

__all__ = [
    "BasicRNNDecoderOutput", "AttentionRNNDecoderOutput",
    "BasicRNNDecoder", "AttentionRNNDecoder"
]

class BasicRNNDecoderOutput(
        collections.namedtuple("BasicRNNDecoderOutput",
                               ("logits", "sample_id", "cell_output"))):
    """The outputs of basic RNN decoders that include both RNN results and
    sampled ids at each step.

    Attributes:
        logits: The output of RNN at each step by applying the
            output layer on cell outputs. E.g., in
            :class:`~texar.modules.BasicRNNDecoder` with default
            hyperparameters, this is a Tensor of
            shape `[batch_size, max_time, vocab_size]`.
        sample_id: The sampled results at each step. E.g., in
            :class:`~texar.modules.BasicRNNDecoder`
            with default helpers (e.g.,
            :class:`~texar.modules.EmbeddingTrainingHelper`), this is a Tensor
            of shape `[batch_size, max_time]` containing the sampled token
            index at each step.
        cell_output: The output of RNN cell at each step. This is the results
            prior to the output layer. E.g., in
            :class:`~texar.modules.BasicRNNDecoder` with default
            hyperparameters, this is a Tensor of
            shape `[batch_size, max_time, cell_output_size]`.
    """
    pass

class AttentionRNNDecoderOutput(
        collections.namedtuple(
            "AttentionRNNDecoderOutput",
            ["logits", "sample_id", "cell_output",
             "attention_scores", "attention_context"])):
    """The outputs of attention RNN decoders that additionally include attention
    results.

    Attributes:
        logits: The outputs of RNN at each step by applying the
            output layer on cell outputs. E.g., in
            :class:`~texar.modules.AttentionRNNDecoder`, this is a Tensor of
            shape `[batch_size, max_time, vocab_size]`.
        sample_id: The sampled results at each step. E.g., in
            :class:`~texar.modules.AttentionRNNDecoder` with default helpers
            (e.g., :class:`~texar.modules.EmbeddingTrainingHelper`), this
            is a Tensor of shape `[batch_size, max_time]` containing the
            sampled token index at each step.
        cell_output: The output of RNN cell at each step. E.g., in
            :class:`~texar.modules.AttentionRNNDecoder`, this is a Tensor of
            shape `[batch_size, max_time, cell_output_size]`.
        attention_scores: A single or tuple of `Tensor`(s) containing the
            alignments emitted at the previous time step for each attention
            mechanism.
        attention_context: The attention emitted at the previous time step.
    """
    pass


class BasicRNNDecoder(RNNDecoderBase):
    """Basic RNN decoder that performs sampling at each step.

    Args:
        cell (RNNCell, optional): An instance of `RNNCell`. If `None`
            (default), a cell is created as specified in
            :attr:`hparams["rnn_cell"]` (see
            :meth:`~texar.modules.BasicRNNDecoder.default_hparams`).
        embedding (optional): A `Variable` or a 2D Tensor (or array)
            of shape `[vocab_size, embedding_dim]` that contains the token
            embeddings.

            Ignore if :attr:`hparams["use_embedding"]` is `False`. Otherwise:

            - If a `Variable`, this is directly used in decoding.

            - If a Tensor or array, a new `Variable` of token embedding is
              created using it as initialization value.

            - If `None` (default), a new `Variable` is created as specified in
              :attr:`hparams["embedding"]`.

        vocab_size (int, optional): Vocabulary size. Required if
            :attr:`hparams["use_embedding"]` is `False` or :attr:`embedding` is
            not provided.
        output_layer (optional): An instance of
            :tf_main:`tf.layers.Layer <layers/Layer>`, or
            :tf_main:`tf.identity <identity>`. Apply to the RNN cell
            output to get logits. If `None`, a dense layer
            is used with output dimension set to :attr:`vocab_size`
            if :attr:`vocab_size` is specified, otherwise inferred from
            from :attr:`embedding` (if embedding is used and
            :attr:`embedding` is specified). Set `output_layer=tf.identity` if
            you do not want to have an output layer after the RNN cell outputs.
        hparams (dict, optional): Hyperparameters. If not specified, the default
            hyperparameter setting is used. See
            :meth:`~texar.modules.BasicRNNDecoder.default_hparams` for the
            structure and default values.
    """

    def __init__(self,
                 cell=None,
                 embedding=None,
                 vocab_size=None,
                 output_layer=None,
                 hparams=None):
        RNNDecoderBase.__init__(self, cell, embedding, vocab_size,
                                output_layer, hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            .. code-block:: python

                {
                    "rnn_cell": default_rnn_cell_hparams(),
                    "use_embedding": True,
                    "embedding": default_embedding_hparams(),
                    "helper_train": default_helper_train_hparams(),
                    "helper_infer": default_helper_infer_hparams(),
                    "max_decoding_length_train": None,
                    "max_decoding_length_infer": None,
                    "name": "basic_rnn_decoder"
                }

            Here:

            "rnn_cell" : dict
                A dictionary of RNN cell hyperparameters. Ignored if
                :attr:`cell` is given when constructing the decoder.

                The default value is defined in
                :meth:`~texar.core.layers.default_rnn_cell_hparams`.

            "use_embedding" : bool
                Whether token embedding is used.

                The default value is `True`.

            "embedding" : dict
                A dictionary of token embedding hyperparameters for
                embedding initialization.

                Ignored if :attr:`embedding` is given and is `Variable`
                when constructing the decoder.

                If :attr:`embedding` is given and is a Tensor or array, the
                "dim" and "initializer" specs of "embedding" are ignored.

                The default value is defined in
                :meth:`~texar.core.layers.default_embedding_hparams`.

            "helper_train" : dict
                A dictionary of :class:`Helper` hyperparameters. The
                helper is used in training phase.

                The default value is defined in
                :meth:`~texar.modules.default_helper_train_hparams`

            "helper_infer": dict
                A dictionary of :class:`Helper` hyperparameters. The
                helper is used in inference phase.

                The default value is defined in
                :meth:`~texar.modules.default_helper_infer_hparams`

            "max_decoding_length_train": int or None
                Maximum allowed number of decoding steps in training phase.

                The default value is `None`, which means decoding is
                performed until fully done, e.g., encountering the <EOS> token.

            "max_decoding_length_infer" : int or None
                Maximum allowed number of decoding steps in inference phase.

                The default value is `None`, which means decoding is
                performed until fully done, e.g., encountering the <EOS> token.

            "name" : str
                Name of the decoder.

                The default value is "basic_rnn_decoder".
        """
        hparams = RNNDecoderBase.default_hparams()
        hparams["name"] = "basic_rnn_decoder"
        return hparams

    def initialize(self, name=None):
        return self._helper.initialize() + (self._initial_state,)

    def step(self, time, inputs, state, name=None):
        cell_outputs, cell_state = self._cell(inputs, state)
        logits = self._output_layer(cell_outputs)
        sample_ids = self._helper.sample(
            time=time, outputs=logits, state=cell_state)
        (finished, next_inputs, next_state) = self._helper.next_inputs(
            time=time,
            outputs=logits,
            state=cell_state,
            sample_ids=sample_ids)
        outputs = BasicRNNDecoderOutput(logits, sample_ids, cell_outputs)
        return (outputs, next_state, next_inputs, finished)

    def finalize(self, outputs, final_state, sequence_lengths):
        return outputs, final_state

    @property
    def output_size(self):
        """Output size of one step.
        """
        return BasicRNNDecoderOutput(
            logits=self._rnn_output_size(),
            sample_id=self._helper.sample_ids_shape,
            cell_output=self._cell.output_size)

    @property
    def output_dtype(self):
        """Types of output of one step.
        """
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and the sample_ids_dtype from the helper.
        dtype = nest.flatten(self._initial_state)[0].dtype
        return BasicRNNDecoderOutput(
            logits=nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            sample_id=self._helper.sample_ids_dtype,
            cell_output=nest.map_structure(
                lambda _: dtype, self._cell.output_size))


#TODO(zhiting): allow a list of Attention Mechanisms
class AttentionRNNDecoder(RNNDecoderBase):
    """RNN decoder with attention mechanism.

    Common arguments are the same as in
    :class:`~texar.modules.BasicRNNDecoder`, including
    :attr:`cell`, :attr:`embedding`, :attr:`vocab_size`,
    and :attr:`output_layer`.

    Args:
        memory: The memory to query; usually the output of an RNN encoder.  This
            tensor should be shaped `[batch_size, max_time, ...]`.
        memory_sequence_length (optional): Sequence lengths for the batch
            entries in memory.  If provided, the memory tensor rows are masked
            with zeros for values past the respective sequence lengths.
        cell_input_fn (callable, optional): A callable that produces RNN cell
            inputs. If `None` (default), the default is used:
            `lambda inputs, attention: tf.concat([inputs, attention], -1)`,
            which cancats regular RNN cell inputs with attentions.
        cell (RNNCell, optional): An instance of `RNNCell`. If `None`
            (default), a cell is created as specified in
            :attr:`hparams["rnn_cell"]` (see
            :meth:`~texar.modules.AttentionRNNDecoder.default_hparams`).
        embedding (optional): A `Variable` or a 2D Tensor (or array)
            of shape `[vocab_size, embedding_dim]` that contains the token
            embeddings.

            Ignore if :attr:`hparams["use_embedding"]` is `False`. Otherwise:

            - If a `Variable`, this is directly used in decoding.

            - If a Tensor or array, a new `Variable` of token embedding is
              created using it as initialization value.

            - If `None` (default), a new `Variable` is created as specified in
              :attr:`hparams["embedding"]`.

        vocab_size (int, optional): Vocabulary size. Required if
            :attr:`hparams["use_embedding"]` is `False` or :attr:`embedding` is
            not provided.
        output_layer (optional): An instance of
            :tf_main:`tf.layers.Layer <layers/Layer>`, or
            :tf_main:`tf.identity <identity>`. Apply to the RNN cell
            output to get logits. If `None`, a dense layer
            is used with output dimension set to :attr:`vocab_size`
            if :attr:`vocab_size` is specified, otherwise inferred from
            from :attr:`embedding` (if embedding is used and
            :attr:`embedding` is specified). Set `output_layer=tf.identity` if
            you do not want to have an output layer after the RNN cell outputs.
        hparams (dict, optional): Hyperparameters. If not specified, the default
            hyperparameter setting is used. See
            :meth:`~texar.modules.AttentionRNNDecoder.default_hparams` for the
            structure and default values.

    """
    def __init__(self,
                 memory,
                 memory_sequence_length=None,
                 cell_input_fn=None,
                 cell=None,
                 embedding=None,
                 vocab_size=None,
                 output_layer=None,
                 hparams=None):
        RNNDecoderBase.__init__(self, cell, embedding, vocab_size,
                                output_layer, hparams)

        attn_hparams = self._hparams['attention']
        attn_kwargs = attn_hparams['kwargs'].todict()

        # Parse the 'probability_fn' argument
        prob_fn = attn_kwargs.get('probability_fn', None)
        if prob_fn is None:
            prob_fn = tf.nn.softmax
        if not callable(prob_fn):
            prob_fn = get_class(prob_fn)
        attn_kwargs['probability_fn'] = prob_fn

        attn_kwargs.update({
            "memory_sequence_length": memory_sequence_length,
            "memory": memory})
        attn_modules = ['tensorflow.contrib.seq2seq', 'texar.custom']
        # Use variable_scope to ensure all trainable variables created in
        # the attention mechanism are collected
        with tf.variable_scope(self.variable_scope):
            attention_mechanism = get_instance(
                attn_hparams["type"], attn_kwargs, attn_modules)

        attn_cell_kwargs = {
            "attention_layer_size": attn_hparams["attention_layer_size"],
            "alignment_history": attn_hparams["alignment_history"],
            "output_attention": attn_hparams["output_attention"],
        }
        # Use variable_scope to ensure all trainable variables created in
        # AttentionWrapper are collected
        with tf.variable_scope(self.variable_scope):
            attn_cell = AttentionWrapper(
                self._cell,
                attention_mechanism,
                cell_input_fn=cell_input_fn,
                **attn_cell_kwargs)
            self._cell = attn_cell

    #TODO(zhiting): fix the TODOs in the docstring
    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values:

        Common hyperparameters are the same as in
        :class:`~texar.modules.BasicRNNDecoder`
        (see :meth:`texar.modules.BasicRNNDecoder.default_hparams`).
        Additional hyperparameters are included for attention mechanism
        configuration.

        Returns:
            .. code-block:: python

                {
                    "attention": {
                        "type": "LuongAttention",
                        "kwargs": {
                            "num_units": 64,
                            "probability_fn": None
                        },
                        "attention_layer_size": None,
                        "alignment_history": False,
                        "output_attention": True,
                    },
                    # The following hyperparameters are common with
                    # `BasicRNNDecoder`
                    "rnn_cell": default_rnn_cell_hparams(),
                    "use_embedding": True,
                    "embedding": default_embedding_hparams(),
                    "helper_train": default_helper_train_hparams(),
                    "helper_infer": default_helper_infer_hparams(),
                    "max_decoding_length_train": None,
                    "max_decoding_length_infer": None,
                    "name": "attention_rnn_decoder"
                }

            Here:

            "attention" : dict
                A dictionary of attention hyperparameters, which includes:

                "type" : str
                    Name or full path to the attention class which can be

                    - Built-in attentions defined in \
                `tensorflow.contrib.seq2seq`, including \
                :class:`~tensorflow.contrib.seq2seq.LuongAttention`,\
                :class:`~tensorflow.contrib.seq2seq.BahdanauAttention`,\
                :class:`~tensorflow.contrib.seq2seq.BahdanauMonotonicAttention`\
                and \
                :class:`~tensorflow.contrib.seq2seq.LuongMonotonicAttention`.
                    - User-defined attention classes in :mod:`texar.custom`.
                    - External attention classes. Must provide the full path, \
                      e.g., "my_module.MyAttentionClass".

                    The default value is "LuongAttention".

                "kwargs" : dict
                    A dictionary of arguments for constructor of the attention
                    class. Any arguments besides the ones in the default
                    value are allowed, except :attr:`memory` and
                    :attr:`memory_sequence_length` (if exist) which are provided
                    by :attr:`attention_keys` and
                    :attr:`attention_values_length` in the decoder's
                    constructor, respectively.

                    The default value is:

                        .. code-block:: python

                            {
                                "num_units": 64,
                                "probability_fn": None
                            }

                        - :attr:`"num_units"` is the depth of the attention \
                        mechanism.

                        - :attr:`"probability_fn"` is a callable or its name \
                        or full path to that converts the attention score to \
                        probabilities. \
                        If `None` (default), the callable is set to  \
                        :meth:`tf.nn.softmax`. Other \
                        options include :meth:`tf.contrib.seq2seq.hardmax` \
                        and :meth:`tf.contrib.sparsemax.sparsemax`. \
                        Its signature should be: \
                        `probabilities = probability_fn(score)`

                    "attention_layer_size" : int or None
                        The depth of the attention (output) layer. If `None`
                        (default), use the context as attention at each time
                        step. Otherwise, feed the context and cell output into
                        the attention layer to generate attention at each time
                        step.

                        The default value is `None`.

                        TODO(zhiting): what does this mean?

                    "alignment_history": bool
                        whether to store alignment history from all time steps
                        in the final output state.

                        The default value is `False`.

                        TODO(zhiting): what does this mean?

                    "output_attention": bool
                        If `True` (default), the output at each time step is
                        the attention value. This is the behavior of Luong-style
                        attention mechanisms. If `False`, the output at each
                        time step is the output of `cell`.  This is the
                        beahvior of Bhadanau-style attention mechanisms.
                        In both cases, the `attention` tensor is propagated to
                        the next time step via the state and is used there.
                        This flag only controls whether the attention mechanism
                        is propagated up to the next cell in an RNN stack or to
                        the top RNN output.

                        The default value is `True`.

        """
        hparams = RNNDecoderBase.default_hparams()
        hparams["name"] = "attention_rnn_decoder"
        hparams["attention"] = {
            "type": "LuongAttention",
            "kwargs": {
                "num_units": 64,
                "probability_fn": None
            },
            "attention_layer_size": None,
            "alignment_history": False,
            "output_attention": True,
        }
        return hparams

    def initialize(self, name=None):
        helper_init = self._helper.initialize()
        return [helper_init[0], helper_init[1], self._initial_state]

    def step(self, time, inputs, state, name=None):
        wrapper_outputs, wrapper_state = self._cell(inputs, state)
        cell_state = wrapper_state.cell_state
        # Essentisally the same as in BasicRNNDecoder.step
        logits = self._output_layer(wrapper_outputs)
        sample_ids = self._helper.sample(
            time=time, outputs=logits, state=cell_state)
        (finished, next_inputs, next_state) = self._helper.next_inputs(
            time=time,
            outputs=logits,
            state=wrapper_state,
            sample_ids=sample_ids)

        attention_scores = wrapper_state.alignments
        attention_context = wrapper_state.attention
        outputs = AttentionRNNDecoderOutput(
            logits, sample_ids, wrapper_outputs,
            attention_scores, attention_context)

        return (outputs, next_state, next_inputs, finished)

    def finalize(self, outputs, final_state, sequence_lengths):
        return outputs, final_state

    @property
    def output_size(self):
        return AttentionRNNDecoderOutput(
            logits=self._rnn_output_size(),
            sample_id=self._helper.sample_ids_shape,
            cell_output=self._cell.output_size,
            attention_scores=self._cell.state_size.alignments,
            attention_context=self._cell.state_size.attention)

    @property
    def output_dtype(self):
        """Types of output of one step.
        """
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and the sample_ids_dtype from the helper.
        dtype = nest.flatten(self._initial_state)[0].dtype
        return AttentionRNNDecoderOutput(
            logits=nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            sample_id=self._helper.sample_ids_dtype,
            cell_output=nest.map_structure(
                lambda _: dtype, self._cell.output_size),
            attention_scores=nest.map_structure(
                lambda _: dtype, self._cell.state_size.alignments),
            attention_context=nest.map_structure(
                lambda _: dtype, self._cell.state_size.attention))

