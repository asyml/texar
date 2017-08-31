#
"""
Helper functions and classes for embedding processing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import gfile
import numpy as np


def load_word2vec(filename, vocab=None):
    """Loads embeddings in the word2vec binary format which has a header line
    containing the number of vectors and their dimensionality (two integers),
    followed with number-of-vectors lines each of which is formatted as
    '<word-string> <embedding-vector>'.

    Args:
        filename: Path to the embedding file.
        vocab: (optional) A dictionary or a set of vocabulary to read embeddings
            from the file. If not given, all word embeddings will be loaded.

    Returns:
        A tuple `(word_vecs, vector_size)`, where `word_vecs` is a dictionary
        that maps word string to its embedding vector (a 1D numpy array), and
        `vector_size` is the embedding dimension.
    """
    word_vecs = {}
    with gfile.GFile(filename, "rb") as fin:
        header = fin.readline()
        vocab_size, vector_size = [int(s) for s in header.split()]
        binary_len = np.dtype('float32').itemsize * vector_size
        for _ in np.arange(vocab_size):
            chars = []
            while True:
                char = fin.read(1)
                if char == ' ':
                    break
                if char != '\n':
                    chars.append(char)
            word = ''.join(chars)
            if vocab is None or word in vocab:
                word_vecs[word] = np.fromstring(fin.read(binary_len),
                                                dtype='float32')
            else:
                fin.read(binary_len)
    return word_vecs, vector_size

def load_glove(filename, vocab=None):
    """Loads embeddings in the glove text format in which each line is
    '<word-string> <embedding-vector>'. Dimensions of the embedding vector
    are separated with whitespace characters.

    Args:
        filename: Path to the embedding file.
        vocab: (optional) A dictionary or a set of vocabulary to read embeddings
            from the file. If not given, all word embeddings will be loaded.

    Returns:
        A tuple `(word_vecs, vector_size)`, where `word_vecs` is a dictionary
        that maps word string to its embedding vector (a 1D numpy array), and
        `vector_size` is the embedding dimension.
    """
    word_vecs = {}
    vector_size = None
    with gfile.GFile(filename) as fin:
        for line in fin:
            vec = line.strip().split()
            if len(vec) == 0:
                continue
            word, vec = vec[0], vec[1:]
            if vocab is not None and word not in vocab:
                continue
            if vector_size is None:
                vector_size = len(vec)
            elif len(vec) != vector_size:
                raise ValueError("Inconsistent word vector sizes: %d vs %d" %
                                 (vector_size, len(vec)))
            word_vecs[word] = np.array([float(v) for v in vec])
    return word_vecs, vector_size


class Embedding(object):
    """Embedding class that loads embedding vectors from file.
    """

    def __init__(self, filename, vocab=None, read_fn=load_word2vec):
        """Constructs the instance and loads embedding from file.

        Args:
            filename: Path to the embedding file.
            vocab: (optional) A dictionary or a set. The vocabulary to read
                embeddings from the file. If not given, all word embeddings
                will be loaded.
            read_fn: Callable that takes `(filename, vocab)` and returns
                `(word_vecs, vector_size)`
        """
        self._filename = filename
        self._vocab = vocab
        self._read_fn = read_fn

        self._word_vecs, self._vector_size = \
            self._read_fn(self._filename, self._vocab)

    @property
    def word_vecs(self):
        """Returns a dictionary that maps word string to its vector.

        Returns:
            A dictionary.
        """
        return self._word_vecs

    @property
    def vector_size(self):
        """Returns the embedding vector size.

        Returns:
            An integer.
        """
        return self._vector_size

