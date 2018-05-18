"""
End to end memory network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.module_base import ModuleBase
from texar.modules.embedders.embedder_utils import get_embedding
from texar.modules.embedders import WordEmbedder
import texar.utils as utils

import copy

__all__ = [
    'MemNetSingleLayer',
    'MemNetRNNLike'
]

def default_embedder_fn(memory, vocab_size, hparams):
    memory_size = hparams["memory_size"]
    #word_embedder = WordEmbedder(vocab_size=vocab_size,
    #    hparams=hparams["word_embedder"])
    word_embedding = get_embedding(
        hparams["word_embedding"].todict(),
        vocab_size=vocab_size,
        variable_scope="word_embedding")
    embedded_memory = tf.nn.embedding_lookup(word_embedding, memory)
    # temporal embedding
    temporal_embedding = get_embedding(
        hparams["temporal_embedding"].todict(),
        vocab_size=memory_size,
        variable_scope="temporal_embedding")
    return tf.add(embedded_memory, temporal_embedding)

class MemNetSingleLayer(ModuleBase):
    """An A-C layer for memory network.

    Args:
        transformer (optional): The matrix H applied to o at the end.
    """

    def __init__(self, transformer=None, hparams=None):
        ModuleBase.__init__(self, hparams)

        self._transformer = transformer

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            A dictionary with the following structure and values.
        """
        return {
            "name": "memnet_single_layer"
        }
    
    def _build(self, query, Aout, Cout, **kwargs):
        """An A-C operation with memory and query vector.

        Args:
            query (Tensor): A `Tensor`.
            Aout (Tensor): output of A operation. [None, memory_size, dim]
            Cout (Tensor): output of C operation. [None, memory_size, dim]

        Returns:
            A `Tensor` of shape same as the output of output_embedder.
        """
        u = query
        with tf.variable_scope(self.variable_scope):
            a = Aout
            c = Cout
            u = tf.expand_dims(u, axis=2)
            p = tf.matmul(a, u)
            p = tf.transpose(p, perm=[0, 2, 1])
            p = tf.nn.softmax(p)
            o = tf.matmul(p, c)
            o = tf.squeeze(o, axis=[1])
            if self._transformer:
                query = tf.matmul(query, self._transformer)
            u_ = tf.add(o, query)

        if not self._built:
            self._add_internal_trainable_variables()
            if self._transformer:
                self._add_trainable_variable(self._transformer)
            self._built = True

        return u_

class MemNetBase(ModuleBase):
    """Base class of memory network
    """

    def __init__(self, vocab_size, input_embedder_fn, output_embedder_fn,
        query_embedder_fn, hparams=None):
        ModuleBase.__init__(self, hparams)
        self._n_hops = self.hparams.n_hops
        self._dim = self.hparams.dim
        self._reludim = self.hparams.reludim
        self._memory_size = self.hparams.memory_size
        self._vocab_size = vocab_size
        self._input_embedder_fn = input_embedder_fn
        self._output_embedder_fn = output_embedder_fn
        self._query_embedder_fn = query_embedder_fn
        with tf.variable_scope(self.variable_scope):
            if self.hparams.need_transformer:
                self.H = tf.get_variable(
                    name="H", shape=[self._dim, self._dim])
            else:
                self.H = None
            self._final_matrix = tf.transpose(
                get_embedding(self.hparams.final_matrix.todict(),
                    vocab_size=vocab_size,
                    variable_scope=self.variable_scope),
                name="final_matrix")

    @staticmethod
    def default_hparams():
        return {
            "name": "memnet_base",
            "n_hops": 1,
            "dim": 100,
            "reludim": 50,
            "memory_size": 100,
            "need_transformer": False,
            "final_matrix": {
                "name": "final_matrix",
                "dim": 100,
                "dropout": {
                    "keep_prob": 1.0
                }
            },
            "dropout": {
                "keep_prob": 1.0
            },
            "variational": False, # share dropout masks in each hop
        }

    def _build(self, memory, query, **kwargs):
        raise NotImplementedError

class MemNetRNNLike(MemNetBase):
    """An implementation of multi-layer end-to-end memory network
        with RNN-like weight tying described in the paper.
    """

    def __init__(self, vocab_size,
        input_embedder_fn=default_embedder_fn,
        output_embedder_fn=default_embedder_fn,
        query_embedder_fn=None, hparams=None):
        MemNetBase.__init__(self, vocab_size, input_embedder_fn,
            output_embedder_fn, query_embedder_fn, hparams)
        with tf.variable_scope(self.variable_scope):
            if self._query_embedder_fn:
                self.B = tf.make_template(
                    "B",
                    self._query_embedder_fn,
                    vocab_size=self._vocab_size,
                    hparams=self.hparams.B,
                    create_scope_now_=True)
            else:
                self.B = None
            self.A = tf.make_template(
                "A",
                self._input_embedder_fn,
                vocab_size=self._vocab_size,
                hparams=self.hparams.A,
                create_scope_now_=True)
            self.C = tf.make_template(
                "C",
                self._output_embedder_fn,
                vocab_size=self._vocab_size,
                hparams=self.hparams.C,
                create_scope_now_=True)
            self.AC = MemNetSingleLayer(self.H,
                hparams={"name": "AC"})
        

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python
        
            {
                "name": "memnet_rnnlike",
                "n_hops": 1,
                "dim": 100,
                "reludim": 50,
                "memory_size": 100,
                "need_transformer": True,
                "final_matrix": {
                    "name": "final_matrix",
                    "dim": 100,
                    "dropout": {
                        "keep_prob": 1.0,
                    },
                }
                "A": {
                    "memory_size": 100,
                    "word_embedder": {
                        "name": "word_embedder",
                        "dim": 100,
                        "initializer": None, # use default initializer
                        "dropout": {
                            "keep_prob": 1.0,
                        },
                    }
                    "temporal_embedding": {
                        "name": "temporal_embedding",
                        "dim": 100,
                        "dropout": {
                            "keep_prob": 1.0,
                        },
                    }
                }
                "C": {
                    "memory_size": 100,
                    "word_embedder": {
                        "name": "word_embedder",
                        "dim": 100,
                        "initializer": None, # use default initializer
                        "dropout": {
                            "keep_prob": 1.0,
                        },
                    }
                    "temporal_embedding": {
                        "name": "temporal_embedding",
                        "dim": 100,
                        "dropout": {
                            "keep_prob": 1.0,
                        },
                    }
                }
                "B": {
                    "memory_size": 100,
                    "word_embedder": {
                        "name": "word_embedder",
                        "dim": 100,
                        "initializer": None, # use default initializer
                        "dropout": {
                            "keep_prob": 1.0,
                        },
                    }
                    "temporal_embedding": {
                        "name": "temporal_embedding",
                        "dim": 100,
                        "dropout": {
                            "keep_prob": 1.0,
                        },
                    }
                }
                "dropout": { # dropout after each hop
                    "keep_prob": 1.0,
                }
            }
        """
        hparams = MemNetBase.default_hparams()
        hparams["name"] = "memnet_rnnlike"
        hparams["need_transformer"] = True
        default_embedder_hparams = {
            "memory_size": 100,
            "word_embedding": {
                "name": "word_embedding",
                "dim": 100,
                "dropout": {
                    "keep_prob": 1.0
                }
            },
            "temporal_embedding": {
                "name": "temporal_embedding",
                "dim": 100,
                "dropout": {
                    "keep_prob": 1.0
                }
            }
        }
        for _ in ("A", "C", "B"):
            hparams[_] = default_embedder_hparams
        return hparams

    def _build(self, memory, query, **kwargs):
        with tf.variable_scope(self.variable_scope):
            if self._query_embedder_fn:
                query = self.B(query)
            self.u = [query]
            self.Aout = self.A(memory)
            self.Cout = self.C(memory)

            keep_prob = utils.switch_dropout(self.hparams.dropout.keep_prob)
            if self.hparams.variational:
                with tf.variable_scope("variational_dropout"):
                    noise = tf.random_uniform(tf.shape(self.u[-1]))
                    random_tensor = keep_prob + noise
                    binary_tensor = tf.floor(random_tensor)
                def variational_dropout(val):
                    return tf.div(val, keep_prob) * binary_tensor

            for k in range(self._n_hops):
                u_ = self.AC(self.u[-1], self.Aout, self.Cout)
                if self._reludim == 0:
                    pass
                elif self._reludim == self._dim:
                    u_ = tf.nn.relu(u_)
                elif 0 < self._reludim < self._dim:
                    linear_part = u_[:, : self._dim - self._reludim]
                    relu_part = u_[:, self._dim - self._reludim :]
                    relued_part = tf.nn.relu(relu_part)
                    u_ = tf.concat(axis=1, values=[linear_part, relued_part])
                else:
                    raise Exception("reludim = {} is illegal".format(
                        self._reludim))
                if self.hparams.variational:
                    u_ = variational_dropout(u_)
                else:
                    u_ = tf.nn.dropout(u_, keep_prob)
                self.u.append(u_)
            logits = tf.matmul(self.u[-1], self._final_matrix)

        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True
        
        return logits
