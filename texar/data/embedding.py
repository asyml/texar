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
Helper functions and classes for embedding processing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow import gfile
import numpy as np

from texar.utils import utils
from texar.hyperparams import HParams

__all__ = [
    "load_word2vec",
    "load_glove",
    "Embedding"
]

def load_word2vec(filename, vocab, word_vecs):
    """Loads embeddings in the word2vec binary format which has a header line
    containing the number of vectors and their dimensionality (two integers),
    followed with number-of-vectors lines each of which is formatted as
    '<word-string> <embedding-vector>'.

    Args:
        filename (str): Path to the embedding file.
        vocab (dict): A dictionary that maps token strings to integer index.
            Tokens not in :attr:`vocab` are not read.
        word_vecs: A 2D numpy array of shape `[vocab_size, embed_dim]`
            which is updated as reading from the file.

    Returns:
        The updated :attr:`word_vecs`.
    """
    with gfile.GFile(filename, "rb") as fin:
        header = fin.readline()
        vocab_size, vector_size = [int(s) for s in header.split()]
        if vector_size != word_vecs.shape[1]:
            raise ValueError("Inconsistent word vector sizes: %d vs %d" %
                             (vector_size, word_vecs.shape[1]))
        binary_len = np.dtype('float32').itemsize * vector_size
        for _ in np.arange(vocab_size):
            chars = []
            while True:
                char = fin.read(1)
                if char == b' ':
                    break
                if char != b'\n':
                    chars.append(char)
            word = b''.join(chars)
            word = tf.compat.as_text(word)
            if word in vocab:
                word_vecs[vocab[word]] = np.fromstring(
                    fin.read(binary_len), dtype='float32')
            else:
                fin.read(binary_len)
    return word_vecs

def load_glove(filename, vocab, word_vecs):
    """Loads embeddings in the glove text format in which each line is
    '<word-string> <embedding-vector>'. Dimensions of the embedding vector
    are separated with whitespace characters.

    Args:
        filename (str): Path to the embedding file.
        vocab (dict): A dictionary that maps token strings to integer index.
            Tokens not in :attr:`vocab` are not read.
        word_vecs: A 2D numpy array of shape `[vocab_size, embed_dim]`
            which is updated as reading from the file.

    Returns:
        The updated :attr:`word_vecs`.
    """
    with gfile.GFile(filename) as fin:
        for line in fin:
            vec = line.strip().split()
            if len(vec) == 0:
                continue
            word, vec = vec[0], vec[1:]
            word = tf.compat.as_text(word)
            if word not in vocab:
                continue
            if len(vec) != word_vecs.shape[1]:
                raise ValueError("Inconsistent word vector sizes: %d vs %d" %
                                 (len(vec), word_vecs.shape[1]))
            word_vecs[vocab[word]] = np.array([float(v) for v in vec])
    return word_vecs


class Embedding(object):
    """Embedding class that loads token embedding vectors from file. Token
    embeddings not in the embedding file are initialized as specified in
    :attr:`hparams`.

    Args:
        vocab (dict): A dictionary that maps token strings to integer index.
        read_fn: Callable that takes `(filename, vocab, word_vecs)` and
            returns the updated `word_vecs`. E.g.,
            :func:`~texar.data.embedding.load_word2vec` and
            :func:`~texar.data.embedding.load_glove`.
    """
    def __init__(self, vocab, hparams=None):
        self._hparams = HParams(hparams, self.default_hparams())

        # Initialize embeddings
        init_fn_kwargs = self._hparams.init_fn.kwargs.todict()
        if "shape" in init_fn_kwargs or "size" in init_fn_kwargs:
            raise ValueError("Argument 'shape' or 'size' must not be "
                             "specified. They are inferred automatically.")
        init_fn = utils.get_function(
            self._hparams.init_fn.type,
            ["numpy.random", "numpy", "texar.custom"])

        try:
            self._word_vecs = init_fn(size=[len(vocab), self._hparams.dim],
                                      **init_fn_kwargs)
        except TypeError:
            self._word_vecs = init_fn(shape=[len(vocab), self._hparams.dim],
                                      **init_fn_kwargs)

        # Optionally read embeddings from file
        if self._hparams.file is not None and self._hparams.file != "":
            read_fn = utils.get_function(
                self._hparams.read_fn,
                ["texar.data.embedding", "texar.data", "texar.custom"])

            self._word_vecs = \
                read_fn(self._hparams.file, vocab, self._word_vecs)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values:

        .. role:: python(code)
           :language: python

        .. code-block:: python

            {
                "file": "",
                "dim": 50,
                "read_fn": "load_word2vec",
                "init_fn": {
                    "type": "numpy.random.uniform",
                    "kwargs": {
                        "low": -0.1,
                        "high": 0.1,
                    }
                },
            }

        Here:

        "file" : str
            Path to the embedding file. If not provided, all embeddings are
            initialized with the initialization function.

        "dim": int
            Dimension size of each embedding vector

        "read_fn" : str or callable
            Function to read the embedding file. This can be the function,
            or its string name or full module path. E.g.,

            .. code-block:: python

                "read_fn": texar.data.load_word2vec
                "read_fn": "load_word2vec"
                "read_fn": "texar.data.load_word2vec"
                "read_fn": "my_module.my_read_fn"

            If function string name is used, the function must be in
            one of the modules: :mod:`texar.data` or :mod:`texar.custom`.

            The function must have the same signature as with
            :func:`load_word2vec`.

        "init_fn" : dict
            Hyperparameters of the initialization function used to initialize
            embedding of tokens missing in the embedding
            file.

            The function must accept argument named `size` or `shape` to
            specify the output shape, and return a numpy array of the shape.

            The `dict` has the following fields:

                "type" : str or callable
                    The initialization function. Can be either the function,
                    or its string name or full module path.

                "kwargs" : dict
                    Keyword arguments for calling the function. The function
                    is called with :python:`init_fn(size=[.., ..], **kwargs)`.
        """
        return {
            "file": "",
            "dim": 50,
            "read_fn": "load_word2vec",
            "init_fn": {
                "type": "numpy.random.uniform",
                "kwargs": {
                    "low": -0.1,
                    "high": 0.1,
                },
            },
            "@no_typecheck": ["read_fn", "init_fn"]
        }

    @property
    def word_vecs(self):
        """2D numpy array of shape `[vocab_size, embedding_dim]`.
        """
        return self._word_vecs

    @property
    def vector_size(self):
        """The embedding dimention size.
        """
        return self._hparams.dim
