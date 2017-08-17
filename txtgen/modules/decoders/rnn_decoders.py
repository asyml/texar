#
"""
Various RNN decoders
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.seq2seq import BasicDecoderOutput
from tensorflow.python.framework import tensor_shape, dtypes    # pylint: disable=E0611

from txtgen.modules.decoders.rnn_decoder_base import RNNDecoderBase


class BasicRNNDecoder(RNNDecoderBase):
    """Basic RNN decoder that performs simple sampling at each step.
    """

    def __init__(self, vocab_size, cell=None, hparams=None,
                 name="basic_rnn_decoder"):
        RNNDecoderBase.__init__(self, cell, hparams, name)
        self._vocab_size = vocab_size

    def initialize(self, name=None):
        pass

    def step(self, time, inputs, state, name=None):
        pass

    def finalize(self, outputs, final_state, sequence_lengths):
        pass

    @property
    def output_size(self):
        return BasicDecoderOutput(
            rnn_output=self._vocab_size,
            sample_id=tensor_shape.TensorShape([]))

    @property
    def output_dtype(self):
        return BasicDecoderOutput(
            rnn_output=dtypes.float32, sample_id=dtypes.int32)
