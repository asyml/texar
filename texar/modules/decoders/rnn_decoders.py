#
"""
Various RNN decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=no-name-in-module, too-many-arguments, too-many-locals
# pylint: disable=not-context-manager, protected-access, invalid-name

import collections
import copy

import tensorflow as tf
from tensorflow.contrib.seq2seq import AttentionWrapper
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq import tile_batch

from texar.modules.decoders.rnn_decoder_base import RNNDecoderBase
from texar.utils import utils

__all__ = [
    "BasicRNNDecoderOutput",
    "AttentionRNNDecoderOutput",
    "BasicRNNDecoder",
    "AttentionRNNDecoder"
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
            :class:`~texar.modules.TrainingHelper`), this is a Tensor
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
            (e.g., :class:`~texar.modules.TrainingHelper`), this
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
        vocab_size (int, optional): Vocabulary size. Required if
            :attr:`output_layer:` is `None`.
        output_layer (optional): An instance of
            :tf_main:`tf.layers.Layer <layers/Layer>`, or
            :tf_main:`tf.identity <identity>`. Apply to the RNN cell
            output to get logits. If `None`, a dense layer
            is used with output dimension set to :attr:`vocab_size`.
            Set `output_layer=tf.identity` if you do not want to have an
            output layer after the RNN cell outputs.
        cell_dropout_mode (optional): A Tensor taking value of
            :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, which
            toggles dropout in the RNN cell (e.g., activates dropout in the
            TRAIN mode). If `None`, :func:`~texar.context.global_mode` is used.
            Ignored if :attr:`cell` is given.
        hparams (dict, optional): Hyperparameters. If not specified, the default
            hyperparameter setting is used. See
            :meth:`~texar.modules.BasicRNNDecoder.default_hparams` for the
            structure and default values.
    """

    def __init__(self,
                 cell=None,
                 vocab_size=None,
                 output_layer=None,
                 cell_dropout_mode=None,
                 hparams=None):
        RNNDecoderBase.__init__(
            self, cell, vocab_size, output_layer, cell_dropout_mode, hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            .. code-block:: python

                {
                    "rnn_cell": default_rnn_cell_hparams(),
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
                Maximum allowed number of decoding steps in training mode..

                The default is `None`, which means decoding is
                performed until fully done, e.g., encountering the <EOS> token.

            "max_decoding_length_infer" : int or None
                Maximum allowed number of decoding steps in inference mode.

                The default is `None`, which means decoding is
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
    :attr:`cell`, :attr:`vocab_size`, and :attr:`output_layer`.

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
        vocab_size (int, optional): Vocabulary size. Required if
            :attr:`output_layer:` is `None`.
        output_layer (optional): An instance of
            :tf_main:`tf.layers.Layer <layers/Layer>`, or
            :tf_main:`tf.identity <identity>`. Apply to the RNN cell
            output to get logits. If `None`, a dense layer
            is used with output dimension set to :attr:`vocab_size`.
            Set `output_layer=tf.identity` if you do not want to have an
            output layer after the RNN cell outputs.
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
                 vocab_size=None,
                 output_layer=None,
                 cell_dropout_mode=None,
                 hparams=None):
        RNNDecoderBase.__init__(
            self, cell, vocab_size, output_layer, cell_dropout_mode, hparams)

        attn_hparams = self._hparams['attention']
        attn_kwargs = attn_hparams['kwargs'].todict()

        # Parse the 'probability_fn' argument
        if 'probability_fn' in attn_kwargs:
            prob_fn = attn_kwargs['probability_fn']
            if prob_fn is not None and not callable(prob_fn):
                prob_fn = utils.get_function(
                    prob_fn,
                    ['tensorflow.nn', 'tensorflow.contrib.sparsemax',
                     'tensorflow.contrib.seq2seq'])
            attn_kwargs['probability_fn'] = prob_fn

        attn_kwargs.update({
            "memory_sequence_length": memory_sequence_length,
            "memory": memory})
        self._attn_kwargs = attn_kwargs
        attn_modules = ['tensorflow.contrib.seq2seq', 'texar.custom']
        # Use variable_scope to ensure all trainable variables created in
        # the attention mechanism are collected
        with tf.variable_scope(self.variable_scope):
            attention_mechanism = utils.check_or_get_instance(
                attn_hparams["type"], attn_kwargs, attn_modules,
                classtype=tf.contrib.seq2seq.AttentionMechanism)

        self._attn_cell_kwargs = {
            "attention_layer_size": attn_hparams["attention_layer_size"],
            "alignment_history": attn_hparams["alignment_history"],
            "output_attention": attn_hparams["output_attention"],
        }
        self._cell_input_fn = cell_input_fn
        # Use variable_scope to ensure all trainable variables created in
        # AttentionWrapper are collected
        with tf.variable_scope(self.variable_scope):
            attn_cell = AttentionWrapper(
                self._cell,
                attention_mechanism,
                cell_input_fn=self._cell_input_fn,
                **self._attn_cell_kwargs)
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
                            "num_units": 256,
                        },
                        "attention_layer_size": None,
                        "alignment_history": False,
                        "output_attention": True,
                    },
                    # The following hyperparameters are the same with
                    # `BasicRNNDecoder`
                    "rnn_cell": default_rnn_cell_hparams(),
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
                                "num_units": 256,
                            }

                        - :attr:`"num_units"` is the depth of the attention \
                        mechanism.

                        E.g., We can specify :attr:`probability_fn` for
                        :tf_main:`LuongAttention
                        <contrib/seq2seq/LuongAttention>` or
                        :tf_main:`BahdanauAttention
                        <contrib/seq2seq/BahdanauAttention>` like:

                        .. code-block:: python

                            {
                                "probability_fn": pf_value,
                                ...
                            }

                        where :attr:`pf_value` is a callable or its name
                        or full path to that converts the attention score to
                        probabilities.
                        The callable can be :tf_main:`tf.nn.softmax
                        <nn/softmax>` (default),
                        :tf_main:`tf.contrib.seq2seq.hardmax
                        <contrib/seq2seq/hardmax>`, or
                        :tf_main:`tf.contrib.sparsemax.sparsemax
                        <contrib/sparsemax/sparsemax>`.
                        Its signature should be:
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
                "num_units": 256,
            },
            "attention_layer_size": None,
            "alignment_history": False,
            "output_attention": True,
        }
        return hparams

    def _get_beam_search_cell(self, beam_width):
        """Returns the RNN cell for beam search decoding.
        """
        with tf.variable_scope(self.variable_scope, reuse=True):
            attn_kwargs = copy.copy(self._attn_kwargs)

            memory = attn_kwargs['memory']
            attn_kwargs['memory'] = tile_batch(memory, multiplier=beam_width)

            memory_seq_length = attn_kwargs['memory_sequence_length']
            if memory_seq_length is not None:
                attn_kwargs['memory_sequence_length'] = tile_batch(
                    memory_seq_length, beam_width)

            attn_modules = ['tensorflow.contrib.seq2seq', 'texar.custom']
            bs_attention_mechanism = utils.check_or_get_instance(
                self._hparams.attention.type, attn_kwargs, attn_modules,
                classtype=tf.contrib.seq2seq.AttentionMechanism)

            bs_attn_cell = AttentionWrapper(
                self._cell._cell,
                bs_attention_mechanism,
                cell_input_fn=self._cell_input_fn,
                **self._attn_cell_kwargs)

            self._beam_search_cell = bs_attn_cell

            return bs_attn_cell

    def initialize(self, name=None):
        helper_init = self._helper.initialize()

        flat_initial_state = nest.flatten(self._initial_state)
        dtype = flat_initial_state[0].dtype
        initial_state = self._cell.zero_state(
            batch_size=tf.shape(flat_initial_state[0])[0], dtype=dtype)
        initial_state = initial_state.clone(cell_state=self._initial_state)

        return [helper_init[0], helper_init[1], initial_state]

    def step(self, time, inputs, state, name=None):
        wrapper_outputs, wrapper_state = self._cell(inputs, state)
        # Essentisally the same as in BasicRNNDecoder.step()
        logits = self._output_layer(wrapper_outputs)
        sample_ids = self._helper.sample(
            time=time, outputs=logits, state=wrapper_state)
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

    def _alignments_size(self):
        # Reimplementation of the alignments_size of each of
        # AttentionWrapper.attention_mechanisms. The original implementation
        # of `_BaseAttentionMechanism._alignments_size`:
        #
        #    self._alignments_size = (self._keys.shape[1].value or
        #                       array_ops.shape(self._keys)[1])
        #
        # can be `None` when the seq length of encoder outputs are priori
        # unknown.
        alignments_size = []
        for am in self._cell._attention_mechanisms:
            az = (am._keys.shape[1].value or tf.shape(am._keys)[1:-1])
            alignments_size.append(az)
        return self._cell._item_or_tuple(alignments_size)

    @property
    def output_size(self):
        return AttentionRNNDecoderOutput(
            logits=self._rnn_output_size(),
            sample_id=self._helper.sample_ids_shape,
            cell_output=self._cell.output_size,
            attention_scores=self._alignments_size(),
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
                lambda _: dtype, self._alignments_size()),
            attention_context=nest.map_structure(
                lambda _: dtype, self._cell.state_size.attention))

    def zero_state(self, batch_size, dtype):
        """Returns zero state of the basic cell.

        Same as :attr:`decoder.cell._cell.zero_state`.
        """
        return self._cell._cell.zero_state(batch_size=batch_size, dtype=dtype)

    def wrapper_zero_state(self, batch_size, dtype):
        """Returns zero state of the attention-wrapped cell.

        Same as :attr:`decoder.cell.zero_state`.
        """
        return self._cell.zero_state(batch_size=batch_size, dtype=dtype)

    @property
    def state_size(self):
        """The state size of the basic cell.

        Same as :attr:`decoder.cell._cell.state_size`.
        """
        return self._cell._cell.state_size


    @property
    def wrapper_state_size(self):
        """The state size of the attention-wrapped cell.

        Same as :attr:`decoder.cell.state_size`.
        """
        return self._cell.state_size

