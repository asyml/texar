"""
transformer encoders. Multihead-SelfAttention
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.core import layers, attentions
from texar import context
from texar.modules.encoders.encoder_base import EncoderBase
from texar.modules.encoders.position_encoders import SinusoidalPositionEncoder
from texar.modules.networks import FeedForwardNetwork
from texar import utils
class TransformerEncoder(EncoderBase):
    """Base class for all encoder classes.
    Args:
        embedding (optional): A `Variable` or a 2D `Tensor` (or `numpy array`)
            of shape `[vocab_size, embedding_dim]` that contains the token
            embeddings.
            If a `Variable`, it is directly used in encoding, and
            the hyperparameters in :attr:`hparams["embedding"]` is ignored.
            If a `Tensor` or `numpy array`, a new `Variable` is created taking
            :attr:`embedding` as initial value. The :attr:`"initializer"` and
            :attr:`"dim"` hyperparameters in :attr:`hparams["embedding"]` are
            ignored.
            If not given, a new `Variable` is created as specified in
            :attr:`hparams["embedding"]`.
        vocab_size (int, optional): The vocabulary size. Required if
            :attr:`embedding` is not provided.
        hparams (dict, optional): Encoder hyperparameters. If it is not
            specified, the default hyperparameter setting is used. See
            :attr:`default_hparams` for the sturcture and default values.
    """
    def __init__(self,
                 embedding,
                 vocab_size=None,
                 hparams=None):
        EncoderBase.__init__(self, hparams)

        self._vocab_size = vocab_size
        self._embedding = None
        self.enc = None
        self.position_enc_embedding = None
        if self._hparams.use_embedding:
            if isinstance(embedding, tf.Variable):
                self._embedding = embedding
            embed_dim = self._embedding.get_shape().as_list()[-1]
            if self._hparams.zero_pad: # TODO(zhiting): vocab has zero pad
                if not self._hparams.bos_pad:
                    self._embedding = tf.concat((tf.zeros(shape=[1, embed_dim]),
                        self._embedding[1:, :]), 0)
                else:
                    self._embedding = tf.concat((tf.zeros(shape=[2, embed_dim]),
                        self._embedding[2:, :]), 0)
            if self._vocab_size is None:
                self._vocab_size = self._embedding.get_shape().as_list()[0]
        with tf.variable_scope(self.variable_scope):
            if self._hparams.target_space_id is not None:
                space_embedding = tf.get_variable('target_space_embedding', \
                    [32, embed_dim])
                self.target_symbol_embedding = tf.gather(space_embedding, \
                    self._hparams.target_space_id)
            self.position_encoder = SinusoidalPositionEncoder()

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        The dictionary has the following structure and default values.
        See :meth:`~texar.core.layers.default_rnn_cell_hparams` for the
        default rnn cell hyperparameters, and
        :meth:`~texar.core.layers.default_embedding_hparams` for the default
        embedding hyperparameters.
        .. code-block:: python
            {
                # (bool) Wether embedding is used in the encoder. If `True`
                # (default), input to the encoder should contain integer
                # indexes and will be used to look up the embedding vectors.
                # If `False`, the input is directly fed into the RNN to encode.
                "use_embedding": True,

                # A dictionary of token embedding hyperparameters for embedding
                # initialization.
                #
                # Ignored if "use_embedding" is `False`, or a tf.Variable
                # is given to `embedding` in the encoder constructor. Note that
                # in the second case, the embedding variable might be updated
                # outside the encoder even if "embedding.trainable" is set to
                # `False` and not updated by the encoder.
                #
                # If a Tensor or array is given to `embedding` in the
                # constructor, "dim" and "initializer" in the configuration
                # are ignored.
                "embedding": texar.core.layers.default_embedding_hparams(),
                # Name of the encoder.
                "name": "transformer_encoder"
            }
        """
        return {
            'multiply_embedding_mode': 'sqrt_depth',
            "use_embedding": True,
            "name":"encoder",
            "zero_pad":True,
            "bos_pad":True,
            #https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
            #Line678
            "max_seq_length":100000000,
            'sinusoid':True,
            'dropout':0.1,
            'num_blocks':6,
            'num_heads':8,
            'poswise_feedforward':None,
            'target_space_id': 1,
            'num_units': 512,
        }

    def _build(self, inputs,  **kwargs):
        self.enc = tf.nn.embedding_lookup(self._embedding, inputs)
        if self._hparams.multiply_embedding_mode =='sqrt_depth':
            self.enc = self.enc * (self._embedding.shape.as_list()[-1]**0.5)

        #### transformer_prepare_encoder
        encoder_padding = utils.embedding_to_padding(self.enc)
        ignore_padding = attentions.attention_bias_ignore_padding(encoder_padding)
        encoder_self_attention_bias = ignore_padding
        encoder_decoder_attention_bias = ignore_padding
        emb_target_space = tf.reshape(self.target_symbol_embedding, [1,1,-1])
        self.enc = self.enc + emb_target_space
        self.enc = self.position_encoder(self.enc, sequence_length=None)
        self.enc = tf.layers.dropout(self.enc, \
            rate=self._hparams.dropout, training=context.global_mode_train())
        pad_remover = utils.padding_related.PadRemover(encoder_padding)
        for i in range(self._hparams.num_blocks):
            with tf.variable_scope("layer_{}".format(i)):
                with tf.variable_scope('self_attention'):
                    selfatt_output = layers.multihead_attention(
                        queries=layers.layer_normalize(self.enc),
                        memory=None,
                        bias=encoder_self_attention_bias,
                        num_heads=self._hparams.num_heads,
                        dropout_rate=self._hparams.dropout,
                        num_units=self._hparams.num_units,
                        scope='multihead_attention'
                    )
                    self.enc = self.enc + tf.layers.dropout(
                        selfatt_output,
                        rate=self._hparams.dropout,
                        training=context.global_mode_train()
                    )
                poswise_network = FeedForwardNetwork(hparams=self._hparams['poswise_feedforward'])
                with tf.variable_scope(poswise_network.variable_scope):
                    x = layers.layer_normalize(self.enc)
                    original_shape = layers.shape_list(x)
                    x = tf.reshape(x, tf.concat([[-1], original_shape[2:]], axis=0))
                    x = tf.expand_dims(pad_remover.remove(x), axis=0)
                    #[1, batch_size*seq_length, hidden_dim]
                    sub_output = tf.layers.dropout(
                        poswise_network(x),
                        rate=self._hparams.dropout,
                        training=context.global_mode_train()
                    )
                    sub_output = tf.reshape(pad_remover.restore(tf.squeeze(\
                        sub_output, axis=0)), original_shape
                    )
                    self.enc = self.enc + sub_output
        self.enc = layers.layer_normalize(self.enc)
        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

        return self.enc, encoder_decoder_attention_bias
