"""
 transformer decoders. Attention is all you need.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
# pylint: disable=no-name-in-module, too-many-arguments, too-many-locals
# pylint: disable=not-context-manager
import tensorflow as tf
from tensorflow.python.framework import tensor_shape, dtypes

from txtgen.modules.module_base import ModuleBase
from txtgen.core import layers
from txtgen import context

class TransformerDecoderOutput(
        collections.namedtuple("TransformerDecoderOutput",
            ("output_logits", "sample_ids"))):
    """the output logits and sampled_ids"""
    pass

class TransformerDecoder(ModuleBase):
    """decoder for transformer: Attention is all you need
    """
    def __init__(self,
                embedding=None,
                vocab_size=None,
                hparams=None):
        ModuleBase.__init__(self, hparams)
        self._embedding = None
        if self._hparams.embedding_enabled:
            if embedding is None and vocab_size == None:
                raise ValueError("If 'embedding' is not provided, "
                        "'vocab_size' must be specified.")
            if isinstance(embedding, tf.Variable):
                self._embedding = embedding
            else:
                self._embedding = layers.get_embedding(
                    self._hparams.embedding, embedding, vocab_size,
                    self.variable_scope)
            embed_dim = self._embedding.shape.as_list()[1]
            if self._hparams.zero_pad:
                self._embedding = tf.concat((tf.zeros(shape=[1, embed_dim]),
                    self._embedding[1:, :]), 0)
            if self._hparams.embedding.trainable:
                self._add_trainable_variable(self._embedding)

    @staticmethod
    def default_hparams():
        return {
            "embedding_enabled": True,
            "embedding": layers.default_embedding_hparams(),
            "name":"transformer_decoder",
            "num_heads":8,
            "num_units":64,
            "zero_pad": True,
        }

    def initialize(self, name=None):
        return self._helper.initialize() + (self._initial_state,)

    def _build(self, inputs, encoder_output):
#        max_decoding_length_train = self._hparams.max_decoding_length_train
#        if max_decoding_length_train is None:
#            max_decoding_length_train = utils.MAX_SEQ_LENGTH
#        if max_decoding_length_infer is None:
#            max_decoding_length_infer = utils.MAX_SEQ_LENGTH
#        max_decoding_length = tf.cond(
#                context.is_train(),
#                lambda: max_decoding_length_train,
#                lambda: max_decoding_length_infer)
        if self._embedding  is not None:
            tgt_embedding = tf.nn.embedding_lookup(self._embedding, inputs)
        else:
            tgt_embedding = inputs
        num_units = tf.shape(tgt_embedding).as_list()[2]
        if self.scale:
            tgt_embedding = tgt_embedding * tf.sqrt(num_units)
        if self._hparams.sinusoid:
            position_dec_embeds = layers.sinusoid_positional_encoding(tgt_embedding,
                    scope = "dec_pe")
        dec_input = tf.layers.dropout(tgt_embedding + position_dec_embeds,
                rate = self._hparams.encoder.dropout_rate,
                training = context.is_train())
        hparams = self._hparams
        with tf.variable_scope(self.variable_scope):
            for i in range(self._hparams.decoder.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    attended_dec = layers.multihead_attention(queries = dec_input,
                            keys = dec_input,
                            num_units = hparams.hidden_units,
                            num_heads = hparams.num_heads,
                            is_training = context.is_train(),
                            dropout_rate = hparams.dropout_rate,
                            causility = True,
                            scope = "self_attention")
                    attended_dec = layers.normalize(attended_dec)

                    attended_dec = layers.multihead_attention(queries = dec_input,
                            keys = encoder_output,
                            num_units = hparams.hidden_units,
                            num_heads = hparams.num_heads,
                            dropout_rate = hparams.dropout_rate,
                            is_training = context.is_train(),
                            causality = False,
                            scope = "vanilla_attention")
                    attended_dec = layers.normalize(attended_dec)

                    attended_dec = layers.poswise_feed_forward(attended_dec)
                    attended_dec = layers.normalize(attended_dec)
                    #[batch, seq_len, hidden_units]
        self.logits = tf.layers.dense(self.attended_dec, self._vocab_size)
        self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))
        #[batch, seq_len]
        return self.logits, self.preds

    @property
    def output_size(self):
        return TransformerDecoderOutput(
            output_logits=tensor_shape.TensorShape([None, None, self._vocab_size]),
            sample_id=tensor_shape.TensorShape([None, None])
            )

    @property
    def output_dtype(self):
        return TransformerDecoderOutput(
            output_logits=dtypes.float32, sample_id=dtypes.int32)

