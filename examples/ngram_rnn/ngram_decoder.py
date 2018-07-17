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
from tensorflow.python.ops import math_ops

from texar.modules.decoders.rnn_decoder_base import RNNDecoderBase
from texar.utils import utils
from texar.modules import BasicRNNDecoder


class NGramRNNDecoderOutput(
        collections.namedtuple(
            "NGramRNNDecoderOutput",
            ("logits", "sample_id",
             'sample_ids0', 'logits_f1',
             'sample_ids1', 'logits_f2',
             'sample_ids2', 'logits_f3',
             "cell_output"))):
    pass


class NGramRNNDecoder(BasicRNNDecoder):
    def __init__(self,
                 cell=None,
                 cell_dropout_mode=None,
                 vocab_size=None,
                 output_layer=None,
                 embedding=None,
                 hparams=None):
        BasicRNNDecoder.__init__(self, cell, cell_dropout_mode,
                                 vocab_size, output_layer, hparams)
        self._embedding = embedding
        with tf.variable_scope(self.variable_scope):
            self._next_k = self._hparams.next_k
            self._output_layer_f1 = self.output_layer
            self._output_layer_f2 = self.output_layer
            self._output_layer_f3 = self.output_layer

    @staticmethod
    def default_hparams():
        hparams = BasicRNNDecoder.default_hparams()
        hparams["name"] = "ngram_rnn_decoder"
        hparams['next_k'] = 1
        return hparams

    def sample_f(self, logits):
        return math_ops.cast(math_ops.argmax(logits, axis=-1), tf.int32)

    def sample_next(self, logits, states, output_layer):
        sample_ids = self.sample_f(logits)
        inputs_f = self._embedding(sample_ids)
        outputs_f, states_f = self._cell(inputs_f, states)
        logits_f = output_layer(outputs_f)
        return sample_ids, logits_f, states_f

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

        if self._next_k >= 2:
            sample_ids0, logits_f1, states_f1 = \
                self.sample_next(logits, cell_state, self._output_layer_f1)
        else:
            sample_ids0, logits_f1, states_f1 = \
                tf.zeros_like(sample_ids), tf.zeros_like(logits), None

        if self._next_k >= 3:
            sample_ids1, logits_f2, states_f2 = \
                self.sample_next(logits_f1, states_f1, self._output_layer_f2)
        else:
            sample_ids1, logits_f2, states_f2 = \
                tf.zeros_like(sample_ids), tf.zeros_like(logits), None

        if self._next_k >= 4:
            sample_ids2, logits_f3, states_f3 = \
                self.sample_next(logits_f2, states_f2, self._output_layer_f3)
        else:
            sample_ids2, logits_f3, states_f3 = \
                tf.zeros_like(sample_ids), tf.zeros_like(logits), None

        outputs = NGramRNNDecoderOutput(
            logits, sample_ids,
            sample_ids0, logits_f1,
            sample_ids1, logits_f2,
            sample_ids2, logits_f3,
            cell_outputs)
        return (outputs, next_state, next_inputs, finished)

    @property
    def output_size(self):
        """Output size of one step.
        """
        return NGramRNNDecoderOutput(
            logits=self._rnn_output_size(),
            sample_id=self._helper.sample_ids_shape,
            sample_ids0=self._helper.sample_ids_shape,
            logits_f1=self._rnn_output_size(),
            sample_ids1=self._helper.sample_ids_shape,
            logits_f2=self._rnn_output_size(),
            sample_ids2=self._helper.sample_ids_shape,
            logits_f3=self._rnn_output_size(),
            cell_output=self._cell.output_size)

    @property
    def output_dtype(self):
        """Types of output of one step.
        """
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and the sample_ids_dtype from the helper.
        dtype = nest.flatten(self._initial_state)[0].dtype
        return NGramRNNDecoderOutput(
            logits=nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            sample_id=self._helper.sample_ids_dtype,
            sample_ids0=self._helper.sample_ids_dtype,
            logits_f1=nest.map_structure(
                lambda _: dtype, self._rnn_output_size()),
            sample_ids1=self._helper.sample_ids_dtype,
            logits_f2=nest.map_structure(
                lambda _: dtype, self._rnn_output_size()),
            sample_ids2=self._helper.sample_ids_dtype,
            logits_f3=nest.map_structure(
                lambda _: dtype, self._rnn_output_size()),
            cell_output=nest.map_structure(
                lambda _: dtype, self._cell.output_size))
