# Copyright 2018 The Texar Authors. All Rights Reserved.
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
End-to-end memory network described in
(Sukhbaatar et al.) End-To-End Memory Networks
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.module_base import ModuleBase
from texar.modules.embedders import WordEmbedder, PositionEmbedder
from texar.utils.mode import switch_dropout
from texar.modules.memory.embed_fns import default_memnet_embed_fn_hparams

# pylint: disable=invalid-name, too-many-instance-attributes, too-many-arguments
# pylint: disable=too-many-locals

__all__ = [
    'MemNetBase',
    'MemNetRNNLike',
]

class MemNetSingleLayer(ModuleBase):
    """An A-C layer for memory network.

    Args:
        H (optional): The matrix :attr:`H` multiplied to :attr:`o` at the end.
        hparams (dict or HParams, optional): Memory network single layer
            hyperparameters. If it is not specified, the default hyperparameter
            setting is used. See :attr:`default_hparams` for the structure and
            default values.
    """

    def __init__(self, H=None, hparams=None):
        ModuleBase.__init__(self, hparams)

        self._H = H

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "name": "memnet_single_layer"
            }

        Here:

        "name": str
            Name of the memory network single layer.
        """
        return {
            "name": "memnet_single_layer"
        }

    def _build(self, u, m, c, **kwargs):
        """An A-C operation with memory and query vector.

        Args:
            u (Tensor): The input query `Tensor` of shape `[None, memory_dim]`.
            m (Tensor): Output of A operation. Should be in shape
                `[None, memory_size, memory_dim]`.
            c (Tensor): Output of C operation. Should be in shape
                `[None, memory_size, memory_dim]`.

        Returns:
            A `Tensor` of shape same as :attr:`u`.
        """
        # Input memory representation
        p = tf.matmul(m, tf.expand_dims(u, axis=2))
        p = tf.transpose(p, perm=[0, 2, 1])

        p = tf.nn.softmax(p) # equ. (1)

        # Output memory representation
        o = tf.matmul(p, c) # equ. (2)
        o = tf.squeeze(o, axis=[1])

        if self._H:
            u = tf.matmul(u, self._H) # RNN-like style
        u_ = tf.add(u, o) # u^{k+1} = H u^k + o^k

        if not self._built:
            self._add_internal_trainable_variables()
            if self._H:
                self._add_trainable_variable(self._H)
            self._built = True

        return u_

class MemNetBase(ModuleBase):
    """Base class inherited by all memory network classes.

    Args:
        raw_memory_dim (int): Dimension size of raw memory entries
            (before embedding). For example,
            if a raw memory entry is a word, this is the **vocabulary size**
            (imagine a one-hot representation of word). If a raw memory entry
            is a dense vector, this is the dimension size of the vector.
        input_embed_fn (optional): A callable that embeds raw memory entries
            as inputs.
            This corresponds to the `A` embedding operation in
            (Sukhbaatar et al.)
            If not provided, a default embedding operation is created as
            specified in :attr:`hparams`. See
            :meth:`~texar.modules.MemNetBase.get_default_embed_fn`
            for details.
        output_embed_fn (optional): A callable that embeds raw memory entries
            as outputs.
            This corresponds to the `C` embedding operation in
            (Sukhbaatar et al.)
            If not provided, a default embedding operation is created as
            specified in :attr:`hparams`. See
            :meth:`~texar.modules.MemNetBase.get_default_embed_fn`
            for details.
        query_embed_fn (optional): A callable that embeds query.
            This corresponds to the `B` embedding operation in
            (Sukhbaatar et al.). If not provided and "use_B" is True
            in :attr:`hparams`, a default embedding operation is created as
            specified in :attr:`hparams`. See
            :meth:`~texar.modules.MemNetBase.get_default_embed_fn`
            for details.
            Notice: If you'd like to customize this callable, please follow
            the same number and style of dimensions as in `input_embed_fn` or
            `output_embed_fn`, and assume that the 2nd dimension of its
            input and output (which corresponds to `memory_size`) is 1.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.
    """

    def __init__(self,
                 raw_memory_dim,
                 input_embed_fn=None,
                 output_embed_fn=None,
                 query_embed_fn=None,
                 hparams=None):
        ModuleBase.__init__(self, hparams)

        self._raw_memory_dim = raw_memory_dim

        self._n_hops = self._hparams.n_hops
        self._relu_dim = self._hparams.relu_dim
        self._memory_size = self._hparams.memory_size

        with tf.variable_scope(self.variable_scope):
            self._A, self._C, self._B, self._memory_dim = self._build_embed_fn(
                input_embed_fn, output_embed_fn, query_embed_fn)

            self.H = None
            if self.hparams.use_H:
                self.H = tf.get_variable(
                    name="H", shape=[self._memory_dim, self._memory_dim])

    def _build_embed_fn(self, input_embed_fn, output_embed_fn, query_embed_fn):
        # Optionally creates embed_fn's
        memory_dim = self.hparams.memory_dim
        mdim_A, mdim_C, mdim_B = None, None, None

        A = input_embed_fn
        if input_embed_fn is None:
            A, mdim_A = self.get_default_embed_fn(
                self._memory_size, self._hparams.A)
            memory_dim = mdim_A

        C = output_embed_fn
        if output_embed_fn is None:
            C, mdim_C = self.get_default_embed_fn(
                self._memory_size, self._hparams.C)
            if mdim_A is not None and mdim_A != mdim_C:
                raise ValueError('Embedding config `A` and `C` must have '
                                 'the same output dimension.')
            memory_dim = mdim_C

        B = query_embed_fn
        if query_embed_fn is None and self._hparams.use_B:
            B, mdim_B = self.get_default_embed_fn(1, self._hparams.B)
            if mdim_A is not None and mdim_A != mdim_B:
                raise ValueError('Embedding config `A` and `B` must have '
                                 'the same output dimension.')
            if mdim_C is not None and mdim_C != mdim_B:
                raise ValueError('Embedding config `C` and `B` must have '
                                 'the same output dimension.')
            memory_dim = mdim_B

        return A, C, B, memory_dim


    def get_default_embed_fn(self, memory_size, embed_fn_hparams):
        """Creates a default embedding function. Can be used for A, C, or B
        operation.

        For B operation (i.e., query_embed_fn), :attr:`memory_size` must be 1.

        The function is a combination of both memory embedding and temporal
        embedding, with the combination method specified by "combine_mode" in
        the `embed_fn_hparams`.

        .. role:: python(code)
           :language: python

        Args:
            embed_fn_hparams (dict or HParams): Hyperparameter of the
                embedding function. See
                :func:`~texar.modules.default_memnet_embed_fn` for details.

        Returns:
            A tuple `(embed_fn, memory_dim)`, where

            - **`memory_dim`** is the dimension of memory entry embedding, \
            inferred from :attr:`embed_fn_hparams`.

                - If `combine_mode` == 'add', `memory_dim` is the \
                embedder dimension.
                - If `combine_mode` == 'concat', `memory_dim` is the sum \
                of the memory embedder dimension and the temporal embedder \
                dimension.

            - **`embed_fn`** is an embedding function that takes in memory \
            and returns memory embedding. \
            Specifically, the function has signature \
            :python:`memory_embedding= embed_fn(memory=None, soft_memory=None)`\
            where one of `memory` and `soft_memory` is provided (but not both).

            Args:
                memory: An `int` Tensor of shape
                    `[batch_size, memory_size]`
                    containing memory indexes used for embedding lookup.
                soft_memory: A Tensor of shape
                    `[batch_size, memory_size, raw_memory_dim]`
                    containing soft weights used to mix the embedding vectors.

            Returns:
                A Tensor of shape `[batch_size, memory_size, memory_dim]`
                containing the memory entry embeddings.

        """
        # memory embedder
        embedder = WordEmbedder(
            vocab_size=self._raw_memory_dim,
            hparams=embed_fn_hparams["embedding"]
        )
        # temporal embedder
        temporal_embedder = PositionEmbedder(
            position_size=memory_size,
            hparams=embed_fn_hparams["temporal_embedding"]
        )

        combine = embed_fn_hparams['combine_mode']
        if combine == 'add':
            if embedder.dim != temporal_embedder.dim:
                raise ValueError('`embedding` and `temporal_embedding` must '
                                 'have the same dimension for "add" '
                                 'combination.')
            memory_dim = embedder.dim
        elif combine == 'concat':
            memory_dim = embedder.dim + temporal_embedder.dim

        def _embed_fn(memory, soft_memory, mode=None):
            if memory is None and soft_memory is None:
                raise ValueError(
                    "Either `memory` or `soft_memory` is required.")
            if memory is not None and soft_memory is not None:
                raise ValueError(
                    "Must not specify `memory` and `soft_memory` at the "
                    "same time.")

            embedded_memory = embedder(
                ids=memory, soft_ids=soft_memory, mode=mode)
            temporal_embedded = temporal_embedder(
                sequence_length=tf.constant([memory_size]), mode=mode)
            temporal_embedded = tf.tile(
                temporal_embedded, [tf.shape(embedded_memory)[0], 1, 1])

            if combine == 'add':
                return tf.add(embedded_memory, temporal_embedded)
            elif combine == 'concat':
                return tf.concat([embedded_memory, temporal_embedded], axis=-1)
            else:
                raise ValueError('Unknown combine method: {}'.format(combine))

        return _embed_fn, memory_dim

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "n_hops": 1,
                "memory_dim": 100,
                "relu_dim": 50,
                "memory_size": 100,
                "A": default_embed_fn_hparams,
                "C": default_embed_fn_hparams,
                "B": default_embed_fn_hparams,
                "use_B": False,
                "use_H": False,
                "dropout_rate": 0,
                "variational": False,
                "name": "memnet",
            }

        Here:

        "n_hops" : int
            Number of hops.

        "memory_dim" : int
            Memory dimension, i.e., the dimension size of a memory entry
            embedding. Ignored if at least one of the embedding functions is
            created according to :attr:`hparams`. In this case
            :attr:`memory_dim` is inferred from the created embed_fn.

        "relu_dim" : int
            Number of elements in :attr:`memory_dim` that have relu at the end
            of each hop.
            Should be not less than 0 and not more than :attr`memory_dim`.

        "memory_size" : int
            Number of entries in memory.

            For example, the number of sentences {x_i} in Fig.1(a) of
            (Sukhbaatar et al.) End-To-End Memory Networks.

        "use_B" : bool
            Whether to create the query embedding function. Ignored if
            `query_embed_fn` is given to the constructor.

        "use_H" : bool
            Whether to perform a linear transformation with matrix `H` at
            the end of each A-C layer.

        "dropout_rate" : float
            The dropout rate to apply to the output of each hop. Should
            be between 0 and 1.
            E.g., `dropout_rate=0.1` would drop out 10% of the units.

        "variational" : bool
            Whether to share dropout masks after each hop.
        """
        return {
            "n_hops": 1,
            "memory_dim": 100,
            "relu_dim": 50,
            "memory_size": 100,
            "A": default_memnet_embed_fn_hparams(),
            "C": default_memnet_embed_fn_hparams(),
            "B": default_memnet_embed_fn_hparams(),
            "use_B": False,
            "use_H": False,
            "dropout_rate": 0,
            "variational": False,
            "name": "memnet",
        }

    def _build(self, memory, query, **kwargs):
        raise NotImplementedError

    @property
    def memory_size(self):
        """The memory size.
        """
        return self._memory_size

    @property
    def raw_memory_dim(self):
        """The dimension of memory element (or vocabulary size).
        """
        return self._raw_memory_dim

    @property
    def memory_dim(self):
        """The dimension of embedded memory and all vectors in hops.
        """
        return self._memory_dim


class MemNetRNNLike(MemNetBase):
    """An implementation of multi-layer end-to-end memory network,
    with RNN-like weight tying described in
    (Sukhbaatar et al.) End-To-End Memory Networks .

    See :meth:`~texar.modules.MemNetBase.get_default_embed_fn` for default
    embedding functions. Customized embedding functions must follow
    the same signature.

    Args:
        raw_memory_dim (int): Dimension size of raw memory entries
            (before embedding). For example,
            if a raw memory entry is a word, this is the **vocabulary size**
            (imagine a one-hot representation of word). If a raw memory entry
            is a dense vector, this is the dimension size of the vector.
        input_embed_fn (optional): A callable that embeds raw memory entries
            as inputs.
            This corresponds to the `A` embedding operation in
            (Sukhbaatar et al.)
            If not provided, a default embedding operation is created as
            specified in :attr:`hparams`. See
            :meth:`~texar.modules.MemNetBase.get_default_embed_fn`
            for details.
        output_embed_fn (optional): A callable that embeds raw memory entries
            as outputs.
            This corresponds to the `C` embedding operation in
            (Sukhbaatar et al.)
            If not provided, a default embedding operation is created as
            specified in :attr:`hparams`. See
            :meth:`~texar.modules.MemNetBase.get_default_embed_fn`
            for details.
        query_embed_fn (optional): A callable that embeds query.
            This corresponds to the `B` embedding operation in
            (Sukhbaatar et al.). If not provided and "use_B" is True
            in :attr:`hparams`, a default embedding operation is created as
            specified in :attr:`hparams`. See
            :meth:`~texar.modules.MemNetBase.get_default_embed_fn`
            for details.
            For customized query_embed_fn, note that the function must follow
            the signature of the default embed_fn where `memory_size` must
            be 1.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.
    """

    def __init__(self,
                 raw_memory_dim,
                 input_embed_fn=None,
                 output_embed_fn=None,
                 query_embed_fn=None,
                 hparams=None):
        MemNetBase.__init__(self, raw_memory_dim, input_embed_fn,
                            output_embed_fn, query_embed_fn, hparams)

        with tf.variable_scope(self.variable_scope):
            self._AC = MemNetSingleLayer(
                self.H,
                hparams={"name": "AC"})

            self._W = tf.layers.Dense(
                units=raw_memory_dim,
                use_bias=False,
                name="W")

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "n_hops": 1,
                "memory_dim": 100,
                "relu_dim": 50,
                "memory_size": 100,
                "A": default_embed_fn_hparams,
                "C": default_embed_fn_hparams,
                "B": default_embed_fn_hparams,
                "use_B": False,
                "use_H": True,
                "dropout_rate": 0,
                "variational": False,
                "name": "memnet_rnnlike",
            }

        Here:

        "n_hops" : int
            Number of hops.

        "memory_dim" : int
            Memory dimension, i.e., the dimension size of a memory entry
            embedding. Ignored if at least one of the embedding functions is
            created according to :attr:`hparams`. In this case
            :attr:`memory_dim` is inferred from the created embed_fn.

        "relu_dim" : int
            Number of elements in :attr:`memory_dim` that have relu at the end
            of each hop.
            Should be not less than 0 and not more than :attr`memory_dim`.

        "memory_size" : int
            Number of entries in memory.

            For example, the number of sentences {x_i} in Fig.1(a) of
            (Sukhbaatar et al.) End-To-End Memory Networks.

        "use_B" : bool
            Whether to create the query embedding function. Ignored if
            `query_embed_fn` is given to the constructor.

        "use_H" : bool
            Whether to perform a linear transformation with matrix `H` at
            the end of each A-C layer.

        "dropout_rate" : float
            The dropout rate to apply to the output of each hop. Should
            be between 0 and 1.
            E.g., `dropout_rate=0.1` would drop out 10% of the units.

        "variational" : bool
            Whether to share dropout masks after each hop.
        """
        hparams = MemNetBase.default_hparams()
        hparams.update({
            "use_H": True,
            "name": "memnet_rnnlike"
        })
        return hparams

    def _build(self, memory=None, query=None, soft_memory=None, soft_query=None,
               mode=None, **kwargs):
        """Pass the :attr:`memory` and :attr:`query` through the memory network
        and return the :attr:`logits` after the final matrix.

        Only one of :attr:`memory` and :attr:`soft_memory` can be specified.
        They should not be specified at the same time.

        Args:
            memory (optional): Memory used in A/C operations. By default, it
                should be an integer tensor of shape
                `[batch_size, memory_size]`,
                containing the ids to embed if provided.
            query (optional): Query vectors as the intial input of the memory
                network.
                If you'd like to apply some transformation (e.g., embedding)
                on it before it's fed into the network, please set `use_B` to
                True and add `query_embed_fn` when constructing this instance.
                If `query_embed_fn` is set to
                :meth:`~texar.modules.MemNetBase.get_default_embed_fn`,
                it should be of shape `[batch_size]`.
                If `use_B` is not set, it should be of shape
                `[batch_size, memory_dim]`.
            soft_memory (optional): Soft memory used in A/C operations. By
                default, it should be a tensor of shape
                `[batch_size, memory_size, raw_memory_dim]`,
                containing the weights used to mix the embedding vectors.
                If you'd like to apply a matrix multiplication on the memory,
                this option can also be used.
            soft_query (optional): Query vectors as the intial input of the
                memory network.
                If you'd like to apply some transformation (e.g., embedding)
                on it before it's fed into the network, please set `use_B` to
                True and add `query_embed_fn` when constructing this instance.
                Similar to :attr:`soft_memory`, if `query_embed_fn` is set to
                :meth:`~texar.modules.MemNetBase.get_default_embed_fn`,
                then it must be of shape `[batch_size, raw_memory_dim]`.
                Ignored if `use_B` is not set.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, including
                `TRAIN`, `EVAL`, and `PREDICT`. If `None`, dropout is
                controlled by :func:`texar.global_mode`.
        """
        if self._B is not None:
            def _unsqueeze(x):
                return x if x is None else tf.expand_dims(x, 1)
            query = tf.squeeze(
                self._B(_unsqueeze(query), _unsqueeze(soft_query), mode=mode),
                1)
        self._u = [query]
        self._m = self._A(memory, soft_memory, mode=mode)
        self._c = self._C(memory, soft_memory, mode=mode)

        keep_prob = switch_dropout(1-self.hparams.dropout_rate, mode=mode)
        if self.hparams.variational:
            with tf.variable_scope("variational_dropout"):
                noise = tf.random_uniform(tf.shape(self._u[-1]))
                random_tensor = keep_prob + noise
                binary_tensor = tf.floor(random_tensor)
            def _variational_dropout(val):
                return tf.div(val, keep_prob) * binary_tensor

        for _ in range(self._n_hops):
            u_ = self._AC(self._u[-1], self._m, self._c)
            if self._relu_dim == 0:
                pass
            elif self._relu_dim == self._memory_dim:
                u_ = tf.nn.relu(u_)
            elif 0 < self._relu_dim < self._memory_dim:
                linear_part = u_[:, : self._memory_dim - self._relu_dim]
                relu_part = u_[:, self._memory_dim - self._relu_dim :]
                relued_part = tf.nn.relu(relu_part)
                u_ = tf.concat(axis=1, values=[linear_part, relued_part])
            else:
                raise ValueError(
                    "relu_dim = {} is illegal".format(self._relu_dim))
            if self.hparams.variational:
                u_ = _variational_dropout(u_)
            else:
                u_ = tf.nn.dropout(u_, keep_prob)
            self._u.append(u_)

        logits = self._W(self._u[-1])

        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

        return logits
