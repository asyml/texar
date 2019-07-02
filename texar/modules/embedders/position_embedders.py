# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Various position embedders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

from texar.modules.embedders.embedder_base import EmbedderBase
from texar.modules.embedders import embedder_utils
from texar.utils.mode import is_train_mode
from texar.utils.shapes import mask_sequences
from texar.utils.shapes import shape_list

# pylint: disable=arguments-differ, invalid-name

__all__ = [
    "PositionEmbedder",
    "SinusoidsPositionEmbedder"
]


class PositionEmbedder(EmbedderBase):
    """Simple position embedder that maps position indexes into embeddings
    via lookup.

    Either :attr:`init_value` or :attr:`position_size` is required. If both are
    given, there must be `init_value.shape[0]==position_size`.

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


    .. document private functions
    .. automethod:: _build
    """

    def __init__(self, init_value=None, position_size=None, hparams=None):
        EmbedderBase.__init__(self, hparams=hparams)

        if init_value is None and position_size is None:
            raise ValueError(
                "Either `init_value` or `position_size` is required."
            )

        self._init_parameterized_embedding(
            init_value, position_size, self._hparams
        )

        self._position_size = position_size
        if position_size is None:
            self._position_size = self._num_embeds
        if self._position_size != self._num_embeds:
            raise ValueError(
                "position_size must equal to init_value.shape[0]."
                "Got %d and %d" % (self._position_size, self._num_embeds)
            )

        self._built = True

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
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
                "name": "position_embedder"
            }

        The hyperparameters have the same meaning as those in
        :meth:`texar.modules.WordEmbedder.default_hparams`.
        """
        hparams = embedder_utils.default_embedding_hparams()
        hparams["name"] = "position_embedder"
        return hparams

    def _build(self, positions=None, sequence_length=None, mode=None, **kwargs):
        """Embeds the positions.

        Either :attr:`positions` or :attr:`sequence_length` is required:

            - If both are given, :attr:`sequence_length` is used to mask out \
            embeddings of those time steps beyond the respective sequence \
            lengths.
            - If only :attr:`sequence_length` is given, then positions \
            from `0` to `sequence_length-1` are embedded.

        Args:
            positions (optional): An integer tensor containing the position
                ids to embed.
            sequence_length (optional): An integer tensor of shape
                `[batch_size]`. Time steps beyond
                the respective sequence lengths will have zero-valued
                embeddings.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, including
                `TRAIN`, `EVAL`, and `PREDICT`. If `None`, dropout will be
                controlled by :func:`texar.global_mode`.
            kwargs: Additional keyword arguments for
                :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>` besides
                :attr:`params` and :attr:`ids`.

        Returns:
            A `Tensor` of shape `shape(inputs) + embedding dimension`.
        """
        # Gets embedder inputs
        # pylint:disable=too-many-locals
        inputs = positions
        if positions is None:
            if sequence_length is None:
                raise ValueError(
                    "Either `positions` or `sequence_length` is required."
                )
            max_length = tf.reduce_max(sequence_length)
            single_inputs = tf.range(start=0, limit=max_length, dtype=tf.int32)
            # Expands `single_inputs` to have shape [batch_size, max_length]
            expander = tf.expand_dims(tf.ones_like(sequence_length), -1)
            inputs = expander * tf.expand_dims(single_inputs, 0)
        ids_rank = len(inputs.shape.dims)

        embedding = self._embedding

        is_training = is_train_mode(mode)

        # Gets dropout strategy
        st = self._hparams.dropout_strategy
        if positions is None and st == "item":
            # If `inputs` is based on `sequence_length`, then dropout
            # strategies 'item' and 'item_type' have the same effect, we
            # use 'item_type' to avoid unknown noise_shape in the 'item'
            # strategy
            st = "item_type"

        # Dropouts as 'item_type' before embedding
        if st == "item_type":
            dropout_layer = self._get_dropout_layer(
                self._hparams, dropout_strategy=st
            )
            if dropout_layer:
                embedding = dropout_layer.apply(
                    inputs=embedding, training=is_training
                )

        # Embeds
        outputs = tf.nn.embedding_lookup(embedding, inputs, **kwargs)

        # Dropouts as 'item' or 'elements' after embedding
        if st != "item_type":
            dropout_layer = self._get_dropout_layer(
                self._hparams,
                ids_rank=ids_rank,
                dropout_input=outputs,
                dropout_strategy=st,
            )
            if dropout_layer:
                outputs = dropout_layer.apply(
                    inputs=outputs, training=is_training
                )

        # Optionally masks
        if sequence_length is not None:
            outputs = mask_sequences(
                outputs,
                sequence_length,
                tensor_rank=len(inputs.shape.dims) + self._dim_rank,
            )

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
    via sinusoid calculation. This module does not have trainable parameters.
    Used in, e.g., Transformer models
    `(Vaswani et al.) "Attention Is All You Need"`.

    Each channel of the input Tensor is incremented by a sinusoid of a
    different frequency and phase.
    This allows attention to learn to use absolute and relative positions.

    Timing signals should be added to some precursors of both the query
    and the memory inputs to attention.
    The use of relative position is possible because sin(x+y) and
    cos(x+y) can be experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to dim / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the dim dimension.

    Args:
        position_size (int): The number of possible positions, e.g., the maximum
            sequence length. Set `position_size=None` and
            `hparams['cache_embeddings']=False` to enable infinite large or
             negative position indexes.

    .. document private functions
    .. automethod:: _build
    """

    def __init__(self, position_size, hparams=None):
        EmbedderBase.__init__(self, hparams=hparams)

        self._num_embeds = position_size
        self._dim = self._hparams.dim
        self._cache_embeddings = self._hparams.cache_embeddings

        num_timescales = self._dim // 2
        min_timescale = self._hparams.min_timescale
        max_timescale = self._hparams.max_timescale

        log_timescale_increment = math.log(
            float(max_timescale) / float(min_timescale)
        ) / (tf.to_float(num_timescales) - 1)
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment
        )
        self.inv_timescales = inv_timescales

        if self._cache_embeddings:
            if position_size is None:
                raise ValueError(
                    "'position_size' must not be None when "
                    "'cache_embeddings' is set to True"
                )
            positions = tf.to_float(tf.range(position_size, dtype=tf.int32))
            signal = self._compute_embeddings(positions)
            self.signal = signal

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values
        We use a geometric sequence of timescales starting with
        min_timescale and ending with max_timescale. The number of different
        timescales is equal to dim/2.

        .. code-block:: python

            {
                'min_timescale': 1.0,
                'max_timescale': 10000.0,
                'dim': 512,
                'cache_embeddings': True,
                'name':'sinusoid_posisiton_embedder',
            }

            Here:

            `"cache_embeddings"`: bool
                If `True`, precompute embeddings for positions in range
                `[0, position_size - 1]`. This leads to faster lookup but
                requires lookup indices to be within this range.

                If `False`, embeddings are computed on-the-fly during lookup.
                Set to `False` if your application needs to handle sequences
                of arbitrary length, or requires embeddings at negative
                positions.
        """
        hparams = {
            "min_timescale": 1.0,
            "max_timescale": 1.0e4,
            "dim": 512,
            "cache_embeddings": True,
            "name": "sinusoid_posisiton_embedder",
        }
        return hparams

    def _compute_embeddings(self, positions):
        inv_timescales = self.inv_timescales
        scaled_time = tf.reshape(tf.cast(positions, inv_timescales.dtype),
                                 (-1, 1)) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat(
            [tf.sin(scaled_time), tf.cos(scaled_time)], axis=1
        )
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(self._dim, 2)]])
        signal = tf.reshape(signal, shape_list(positions) + [self._dim])
        return signal

    def _build(self, positions=None, sequence_length=None):
        """Embeds.
        Either :attr:`positions` or :attr:`sequence_length` is required:

            - If both are given, :attr:`sequence_length` is used to mask out \
            embeddings of those time steps beyond the respective sequence \
            lengths.
            - If only :attr:`sequence_length` is given, then positions \
            from `0` to `sequence_length-1` are embedded.

        Args:
            positions (optional): An integer tensor containing the position
                ids to embed.
            sequence_length (optional): An integer tensor of shape
                `[batch_size]`. Time steps beyond
                the respective sequence lengths will have zero-valued
                embeddings.

        Returns:
            A `Tensor` of shape `[batch_size, max_time, dim]`.
        """

        if positions is None:
            if sequence_length is None:
                raise ValueError(
                    "Either `positions` or `sequence_length` is required."
                )
            max_length = tf.reduce_max(sequence_length)
            single_inputs = tf.range(start=0, limit=max_length, dtype=tf.int32)
            # Expands `single_inputs` to have shape [batch_size, max_length]
            expander = tf.expand_dims(tf.ones_like(sequence_length), -1)
            inputs = expander * tf.expand_dims(single_inputs, 0)
        else:
            inputs = positions

        if self._cache_embeddings:
            outputs = tf.nn.embedding_lookup(self.signal, inputs)
        else:
            outputs = self._compute_embeddings(inputs)

        return outputs
