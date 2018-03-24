"""
 transformer decoders. Attention is all you need.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=no-name-in-module, too-many-arguments, too-many-locals
# pylint: disable=not-context-manager

import collections

import tensorflow as tf
from tensorflow.python.framework import tensor_shape, dtypes

from texar.core import layers
from texar import context
from texar.module_base import ModuleBase
from texar.modules.networks import FeedForwardNetwork
from texar.modules.embedders import embedder_utils

class TransformerDecoderOutput(
        collections.namedtuple("TransformerDecoderOutput",\
            ("output_logits", "sample_ids"))):
    """the output logits and sampled_ids"""
    pass

class TransformerDecoder(ModuleBase):
    """decoder for transformer: Attention is all you need
    """
    def __init__(self, embedding=None, vocab_size=None, hparams=None):
        ModuleBase.__init__(self, hparams)
        self._vocab_size = vocab_size
        self._embedding = None
        self.position_dec_embedding = None
        self.dec, self.logits, self.preds = None, None, None
        if self._hparams.use_embedding:
            if embedding is None and vocab_size is None:
                raise ValueError("If 'embedding' is not provided, 'vocab_size' must be specified.")
            #print('type embedding:{}'.format(type(embedding)))
            #<class 'tensorflow.python.framework.ops.Tensor'>
            if isinstance(embedding, tf.Tensor):
                self._embedding = embedding
                print('embedding shared between encoder and decoder')
            else:
                self._embedding = embedder_utils.get_embedding(
                    self._hparams.embedding, embedding, vocab_size,
                    variable_scope=self.variable_scope)
                self._embed_dim = self._embedding.get_shape().as_list()[-1]
                if self._hparams.zero_pad:
                    self._embedding = tf.concat((tf.zeros(shape=[1, self._embed_dim]),\
                        self._embedding[1:, :]), 0)
                if self._hparams.embedding.trainable:
                    self._add_trainable_variable(self._embedding)
            if self._vocab_size is None:
                self._vocab_size = self._embedding.get_shape().as_list()[0]
            if self._hparams.share_embed_and_transform:
                self._output_transform = tf.transpose(self._embedding)
    @staticmethod
    def default_hparams():
        return {
            'multiply_embedding_mode': 'sqrt_depth',
            'share_embed_and_transform': True,
            "use_embedding": True,
            "embedding": embedder_utils.default_embedding_hparams(),
            "name":"decoder",
            "num_heads":8,
            "num_blocks":6,
            "zero_pad": True,
            "max_seq_length":10,
            "scale":False,
            "dropout":0.1,
            "sinusoid":True,
            'poswise_feedforward':None,
        }

    def _build(self, inputs, encoder_output, encoder_padding):
        if self._embedding is not None:
            self.dec = tf.nn.embedding_lookup(self._embedding, inputs)
        else:
            self.dec = inputs

        if self._hparams.multiply_embedding_mode == 'sqrt_depth':
            self.dec = self.dec * (self._embedding.shape.as_list()[-1]**0.5)
        if self._hparams.sinusoid:
            self.dec += layers.sinusoid_positional_encoding(
                self.dec,
                variable_scope='dec_pe')
        else:
            self.position_dec_embedding = embedder_utils.get_embedding(
                hparams=self._hparams.embedding,
                vocab_size=self._hparams.max_seq_length,
                variable_scope='dec_pe')
            self.dec += tf.nn.embedding_lookup(self.position_dec_embedding,\
                tf.tile(tf.expand_dims(tf.range(tf.shape(inputs)[1]), 0), [tf.shape(inputs)[0], 1]))

        self.dec = tf.layers.dropout(
            self.dec,
            rate=self._hparams.dropout,
            training=context.is_train()
        )

        for i in range(self._hparams.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                with tf.variable_scope("self_attention"):
                    selfatt_output = layers.multihead_attention(
                        queries=layers.layer_normalize(self.dec),
                        keys=None,
                        keys_padding=None,
                        num_units=self._hparams.embedding.dim,
                        num_heads=self._hparams.num_heads,
                        dropout_rate=self._hparams.dropout,
                        causality=True,
                        scope="self_attention"
                    )
                    # no padding is ever followed by nonpadding,
                    # so causality can cover keys padding
                    self.dec = self.dec + tf.layers.dropout(
                        selfatt_output,
                        rate=self._hparams.dropout,
                        training=context.is_train()
                    )

                with tf.variable_scope('encdec_attention'):
                    encdec_output = layers.multihead_attention(
                        queries=layers.layer_normalize(self.dec),
                        keys=encoder_output,
                        keys_padding=encoder_padding,
                        num_units=self._hparams.embedding.dim,
                        num_heads=self._hparams.num_heads,
                        dropout_rate=self._hparams.dropout,
                        causality=False,
                        scope="multihead_attention"
                    )
                    self.dec = self.dec + tf.layers.dropout(encdec_output, \
                        rate=self._hparams.dropout,
                        training=context.is_train()
                    )
                poswise_network = FeedForwardNetwork(hparams=self._hparams['poswise_feedforward'])
                with tf.variable_scope(poswise_network.variable_scope):
                    sub_output = tf.layers.dropout(
                        poswise_network(layers.layer_normalize(self.dec)),
                        rate=self._hparams.dropout,
                        training=context.is_train()
                    )
                    self.dec = self.dec + sub_output

        self.dec = layers.layer_normalize(self.dec)
        # share the projection weight with word embedding

        if self._hparams.share_embed_and_transform:
            batch_size, length= tf.shape(self.dec)[0], tf.shape(self.dec)[1]
            depth = self.dec.get_shape()[2]
            self.dec = tf.reshape(self.dec, [-1, depth])
            self.logits = tf.matmul(self.dec, self._output_transform)
            self.logits = tf.reshape(self.logits, [batch_size, length, self._vocab_size])
        else:
            self.logits = tf.layers.dense(self.dec, self._vocab_size)

        self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))

        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True
        return self.logits, self.preds

    @property
    def output_size(self):
        """
        The output of the _build function, (logits, preds)
        logits: [batch_size, length, vocab_size]
        preds: [batch_size, length]
        """
        return TransformerDecoderOutput(
            output_logits=tensor_shape.TensorShape([None, None, self._vocab_size]),
            sample_id=tensor_shape.TensorShape([None, None])
            )

    @property
    def output_dtype(self):
        """
        The output dtype of the _build function, (float32, int32)
        """
        return TransformerDecoderOutput(
            output_logits=dtypes.float32, sample_id=dtypes.int32)
