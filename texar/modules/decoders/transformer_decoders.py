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

from texar.modules.module_base import ModuleBase
from texar.core import layers
from texar import context

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
        self._vocab_size = vocab_size
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
            embed_dim = self._embedding.shape.as_list()[-1]
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
                "num_blocks":6,
                "zero_pad": True,
                "max_seq_length":10,
                "scale":True,
                "dropout":0.1,
                "sinusoid":True,
                "helper_infer": {
                    'type':'GreedyEmbeddingHelper',
                },
            }

    def _build(self, inputs, encoder_output):
        if self._embedding  is not None:
            dec = tf.nn.embedding_lookup(self._embedding, inputs)
        else:
            dec = inputs
        dim = dec.shape.as_list()[-1]
        if self._hparams.scale:
            dec = dec * (dim**0.5)

        with tf.variable_scope(self.variable_scope):
            if self._hparams.sinusoid:
                dec += layers.sinusoid_positional_encoding(dec,
                        num_units=dim,
                        max_time=self._hparams.max_seq_length,
                        variable_scope ="dec_pe")
            else:
                position_dec_embedding = layers.get_embedding(
                        hparams=self._hparams.embedding,
                        vocab_size=self._hparams.max_seq_length,
                        variable_scope='dec_pe',
                        )
                dec += tf.nn.embedding_lookup(position_dec_embedding,
                        tf.tile(tf.expand_dims(tf.range(tf.shape(inputs)[1]), 0),\
                                [inputs.shape[0], 1]))
            print('dropout rate:{}'.format(self._hparams.dropout))
            print('decoder num_heads:{}'.format(self._hparams.num_heads))
            dec = tf.layers.dropout(dec,
                    rate = self._hparams.dropout,
                    training = context.is_train())
            for i in range(self._hparams.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    dec = layers.multihead_attention(queries = dec,
                            keys = dec,
                            num_units = self._hparams.embedding.dim,
                            num_heads = self._hparams.num_heads,
                            dropout_rate = self._hparams.dropout,
                            causality = True,
                            scope = "self_attention")
                    dec = layers.multihead_attention(queries = dec,
                            keys = encoder_output,
                            num_units= self._hparams.embedding.dim,
                            num_heads = self._hparams.num_heads,
                            dropout_rate = self._hparams.dropout,
                            causality = False,
                            scope = "vanilla_attention")
                    dec = layers.poswise_feedforward(dec,
                            num_units=[4*self._hparams.embedding.dim, self._hparams.embedding.dim])

        self.logits = tf.layers.dense(dec, self._vocab_size)
        self.preds = tf.to_int64(tf.argmax(self.logits, axis=-1))
        self._add_internal_trainable_variables()
        self._built = True
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

