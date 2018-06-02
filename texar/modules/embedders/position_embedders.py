#
"""
Various position embedders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.modules.embedders.embedder_base import EmbedderBase
from texar.modules.embedders import embedder_utils
from texar.utils import utils
from texar.utils.shapes import get_batch_size, mask_sequences

import math
__all__ = [
    "PositionEmbedder",
    "SinusoidsPositionEmbedder",
]

class PositionEmbedder(EmbedderBase):
    """Simple position embedder that maps position indexes into embeddings
    via lookup.

    Either :attr:`init_value` or :attr:`position_size` is required. If both are
    given, :attr:`init_value.shape[0]` must equal :attr:`position_size`.

    Args:
        init_value (optional): A `Tensor` or numpy array that contains the
            initial value of embeddings. It is typically of shape
            `[position_size, embedding dim]`

            If `None`, embedding is initialized as specified in
            :attr:`hparams["initializer"]`. Otherwise, the
            :attr:`"initializer"` and :attr:`"dim"`
            hyperparameters in :attr:`hparams` are ignored.
        position_size (int, optional): The number of possible positions, e.g.,
            the maximum sequence length. Required if :attr:`init_value` is
            not given.
        hparams (dict, optional): Embedder hyperparameters. If it is not
            specified, the default hyperparameter setting is used. See
            :attr:`default_hparams` for the sturcture and default values.
    """

    def __init__(self, init_value=None, position_size=None, hparams=None):
        EmbedderBase.__init__(self, hparams=hparams)

        if init_value is None and position_size is None:
            raise ValueError(
                "Either `init_value` or `position_size` is required.")

        self._init_parameterized_embedding(init_value, position_size,
                                           self._hparams)

        self._position_size = position_size
        if position_size is None:
            self._position_size = self._num_embeds
        if self._position_size != self._num_embeds:
            raise ValueError(
                'position_size must equal to init_value.shape[0].'
                'Got %d and %d' % (self._position_size, self._num_embeds))

        self._built = True

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            A dictionary with the following structure and values.

            .. code-block:: python

                {
                    "name": "position_embedder",
                    "dim": 100,
                    "initializer": {
                        "type": "random_uniform_initializer",
                        "kwargs": {
                            "minval": -0.1,
                            "maxval": 0.1,
                            "seed": None
                        }
                    },
                    "regularizer": {
                        "type": "L1L2",
                        "kwargs": {
                            "l1": 0.,
                            "l2": 0.
                        }
                    },
                    "dropout_rate": 0,
                    "trainable": True,
                }

            See :func:`~texar.modules.default_embedding_hparams` for more
            details.
        """
        hparams = embedder_utils.default_embedding_hparams()
        hparams["name"] = "position_embedder"
        return hparams

    def _build(self, positions=None, sequence_length=None, mode=None, **kwargs):
        """Embeds with look-up.

        Either :attr:`position` or :attr:`sequence_length` is required. If both
        are given, :attr:`sequence_length` is ignored.

        Args:
            positions (optional): An integer tensor containing the position
                ids to be looked up.
            sequence_length (optional): An integer tensor of shape
                `[batch_size]`. Time steps beyond
                the respective sequence lengths will have zero-valued
                embeddings.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, including
                `TRAIN`, `EVAL`, and `PREDICT`. If `None`, dropout will be
                controlled by :func:`texar.context.global_mode`.
            kwargs: Additional keyword arguments for
                :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>` besides
                :attr:`params` and :attr:`ids`.

        Returns:
            A `Tensor` of shape `shape(inputs) + embedding dimension`.
        """
        inputs = positions
        if positions is None:
            if sequence_length is None:
                raise ValueError(
                    'Either `positions` or `sequence_length` is required.')
            max_length = tf.reduce_max(sequence_length)
            single_inputs = tf.range(start=0, limit=max_length, dtype=tf.int32)
            inputs = tf.tile(tf.expand_dims(single_inputs, 0),
                             [get_batch_size(sequence_length), 1])

        embedding = self._embedding
        dropout_layer = self._get_dropout_layer(self._hparams, inputs)
        if dropout_layer:
            is_training = utils.is_train_mode(mode)
            if self._hparams.dropout_strategy == 'item_type':
                embedding = dropout_layer.apply(
                    inputs=embedding, training=is_training)

        outputs = tf.nn.embedding_lookup(embedding, inputs, **kwargs)

        if dropout_layer and self._hparams.dropout_strategy != 'item_type':
            outputs = dropout_layer.apply(
                inputs=outputs, training=is_training)

        if positions is None:
            outputs = mask_sequences(
                outputs, sequence_length, tensor_rank=2+self._dim_rank)

        return outputs

    @property
    def embedding(self):
        """The embedding tensor.
        """
        return self._embedding

    @property
    def dim(self):
        """The embedding dimension.
        """
        return self._dim

    @property
    def position_size(self):
        """The position size, i.e., maximum number of positions.
        """
        return self._position_size


class SinusoidsPositionEmbedder(EmbedderBase):
    """Sinusoid position embedder that maps position indexes into embeddings
    via sinusoid calculation.
    Args:
        min_timescale: a float
        max_timescale: a float
    """
    def __init__(self, hparams=None):
        EmbedderBase.__init__(self, hparams=hparams)

    # TODO(zhiting): Add docstring explaining what do all the hyperparameters
    # mean?
    def default_hparams(self):
        """returns a dictionary of hyperparameters with default values"""
        hparams = {
            'name':'sinusoid_posisiton_embedder',
            'min_timescale': 1.0,
            'max_timescale': 1.0e4,
            'trainable': False,
        }
        return hparams

    # TODO(zhiting): would it be better to simply return the position embedding,
    # rather than directly add the embedding to inputs?
    def _build(self, x):
        """add positional embedding to the input"""
        length = utils.shape_list(x)[1]
        channels = utils.shape_list(x)[2]
        position_embeddings = self.get_position_embedding(length, channels)
        return x + position_embeddings

    # TODO(zhiting): add docstring; Is this an interface for users? If not,
    # make it private
    def get_position_embedding(self, length, channels):
        position = tf.to_float(tf.range(length))
        num_timescales = channels // 2
        log_timescale_increment = (
            math.log(float(self.max_timescale) / float(self.min_timescale)) /
            (tf.to_float(num_timescales) - 1))
        inv_timescales = self.min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) \
            * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])
        return signal

