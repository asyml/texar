#
"""
Helper functions and classes for embedding processing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import gfile
import numpy as np

from txtgen.core import utils
from txtgen.hyperparams import HParams

def load_word2vec(filename, vocab, word_vecs):
    """Loads embeddings in the word2vec binary format which has a header line
    containing the number of vectors and their dimensionality (two integers),
    followed with number-of-vectors lines each of which is formatted as
    '<word-string> <embedding-vector>'.

    Args:
        filename: Path to the embedding file.
        vocab: A dictionary that maps token strings to integer index. Tokens not
            in `vocab` are not read.
        word_vecs: A 2D numpy array of shape `[vocab_size, embed_dim]`
            which contains the initial embeddings and is updated as reading from
            the file.

    Returns:
        The updated `word_vecs`.
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
                if char == ' ':
                    break
                if char != '\n':
                    chars.append(char)
            word = ''.join(chars)
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
        filename: Path to the embedding file.
        vocab: A dictionary that maps token strings to integer index. Tokens not
            in `vocab` are not read.
        word_vecs: A 2D numpy array of shape `[vocab_size, embed_dim]`
            which contains the initial embeddings and is updated as reading from
            the file.

    Returns:
        The updated `word_vecs`.
    """
    with gfile.GFile(filename) as fin:
        for line in fin:
            vec = line.strip().split()
            if len(vec) == 0:
                continue
            word, vec = vec[0], vec[1:]
            if word not in vocab:
                continue
            if len(vec) != word_vecs.shape[1]:
                raise ValueError("Inconsistent word vector sizes: %d vs %d" %
                                 (len(vec), word_vecs.shape[1]))
            word_vecs[vocab[word]] = np.array([float(v) for v in vec])
    return word_vecs


class Embedding(object):
    """Embedding class that loads token embedding vectors from file.
    """

    def __init__(self, vocab, hparams=None):
        """Loads embeddings from file and initialzes embeddings of tokens not
        in the file.

        Args:
            vocab (dict): A dictionary that maps token strings to integer index.
            read_fn: Callable that takes `(filename, vocab, word_vecs)` and
                returns the updated `word_vecs`. See
                :meth:`~txtgen.data.embedding.load_word2vec` and
                :meth:`~txtgen.data.embedding.load_glove` for examples.
        """
        self._vocab = vocab
        self._hparams = HParams(hparams, self.default_hparams())

        # Initialize embeddings
        init_fn_kwargs = self._hparams.init_fn.kwargs.todict()
        if "shape" in init_fn_kwargs or "size" in init_fn_kwargs:
            raise ValueError("Argument 'shape' or 'size' must not be specified."
                             " It is inferred automatically.")
        init_fn = utils.get_function(self._hparams.init_fn.type,
                                     ["txtgen.custom", "numpy.random", "numpy"])
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
                ["txtgen.custom", "txtgen.data.embedding"])

            self._word_vecs = \
                read_fn(self._hparams.file, self._vocab, self._word_vecs)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        # TODO(zhiting): add more docs
        return {
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

    @property
    def word_vecs(self):
        """Returns a 2D numpy array where the 1st dimension is the word index
        and the 2nd dimension is the embedding vector.
        """
        return self._word_vecs

    @property
    def vector_size(self):
        """Returns the embedding vector size.

        Returns:
            An integer.
        """
        return self._hparams.dim

