"""
transformer encoders. Multihead-SelfAttention
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from txtgen.modules.encoders.encoder_base import EncoderBase
from txtgen.core import layers

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
                 embedding=None,
                 vocab_size=None,
                 hparams=None):
        EncoderBase.__init__(self, hparams)
        self._embedding = None
        if self._hparams.embedding_enabled:
            if embedding is None and vocab_size is None:
                raise ValueError("If `embedding` is not provided, "
                                "`vocab_size` must be specified.")
            if isinstance(embedding, tf.Variable):
                self._embedding = embedding
            else:
                self._embedding = layers.get_embedding(
                    self._hparams.embedding, embedding, vocab_size,
                    self.variable_scope)
            embed_dim = self._embedding.shape()[1]
            if self._hparams.zero_pad:
                self._embedding = tf.concat((tf.zeros(shape=[1, embed_dim]),
                                            self._embedding[1:, :]), 0)
            if self._hparams.embedding.trainable:
                self._add_trainable_variable(self._embedding)

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
                # (bool) Wether embedding is used in the encoder. If `True`
                # (default), input to the encoder should contain integer
                # indexes and will be used to look up the embedding vectors.
                # If `False`, the input is directly fed into the RNN to encode.
                "embedding_enabled": True,

                # A dictionary of token embedding hyperparameters for embedding
                # initialization.
                #
                # Ignored if "embedding_enabled" is `False`, or a tf.Variable
                # is given to `embedding` in the encoder constructor. Note that
                # in the second case, the embedding variable might be updated
                # outside the encoder even if "embedding.trainable" is set to
                # `False` and not updated by the encoder.
                #
                # If a Tensor or array is given to `embedding` in the
                # constructor, "dim" and "initializer" in the configuration
                # are ignored.
                "embedding": txtgen.core.layers.default_embedding_hparams(),
                # Name of the encoder.
                "name": "transformer_encoder"
            }
        """
        return {
            "embedding_enabled": True,
            "embedding": layers.default_embedding_hparams(),
            "name": "transformer_encoder"
        }

        def _build(self, inputs, **kwargs):
            if self._embedding is not None:
                embedded_inputs = tf.nn.embedding_lookup(self._embedding, inputs)
            else:
                embedded_inputs = inputs
            dim = tf.shape(embedded_inputs)[2]
            if self.scale:
                embedded_inputs = embedded_inputs * tf.sqrt(dim)
            if self._hparams.sinusoid:
                position_inputs = sinusoid_positional_encoding(embedded_inputs,
                        scope="enc_pe")
            enc_output = tf.layers.dropout(embedded_inputs + position_inputs,
                    rate = self._hparams.encoder.dropout_rate,
                    training=True)
            for i in range(self._hparams.encoder.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    enc_output = multihead_attention(queries=enc_output,
                            keys= enc_output,
                            num_heads = self._hparams.encoder.num_heads,
                            dropout_rate = self._hparams.encoder.dropout_rate,
                            is_training = True,
                            causality=False)
                    enc_output = feed_layer(enc_output,
                            num_units=[4*hp.hidden_units, hp.hidden_units])
            return enc_output
