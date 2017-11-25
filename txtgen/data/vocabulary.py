#
"""
Helper functions and classes for vocabulary processing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict

import tensorflow as tf
from tensorflow import gfile
import numpy as np
from txtgen.data.constants import BOS_TOKEN, EOS_TOKEN, PADDING_TOKEN, UNK_TOKEN

__all__ = [
    "Vocab"
]

def _make_defaultdict(keys, values, default_value):
    """Creates a python defaultdict.

    Args:
        keys (list): Keys of the dictionary.
        values (list): Values correspond to keys. The two lists :attr:`keys` and
            :attr:`values` must be of the same length.
        default_value: default value returned when key is missing.

    Returns:
        defaultdict: A python `defaultdict` instance that maps keys to values.
    """
    dict_ = defaultdict(lambda: default_value)
    for k, v in zip(keys, values):  # pylint: disable=invalid-name
        dict_[k] = v

    return dict_


class Vocab(object):  # pylint: disable=too-many-instance-attributes
    """Vocabulary class that loads vocabulary from file, and maintains mapping
    tables between token strings and indexes.

    Args:
        filename (str): Path to the vocabulary file where each line contains
            one word. Each word is indexed with its line number (starting from
            0).
        bos_token (str): A special token that will be added to the beginning of
            sequences.
        eos_token (str): A special token that will be added to the end of
            sequences.
        unk_token (str): A special token that will replace all unknown tokens.
        padding_token (str): A special token that is used to do padding,
                                default to be a empty string.
    """

    def __init__(self, filename, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                 unk_token=UNK_TOKEN, padding_token=PADDING_TOKEN):
        self._filename = filename
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._unk_token = unk_token
        self._padding_token = padding_token

        self._id_to_token_map, self._token_to_id_map, \
        self._id_to_token_map_py, self._token_to_id_map_py = \
            self.load(self._filename)

    def load(self, filename):
        """Loads the vocabulary from the file.

        Args:
            filename (str): Path to the vocabulary file.

        Returns:
            A tuple of TF and python mapping tables between word string and
            index, (:attr:`id_to_token_map`, :attr:`token_to_id_map`,
            :attr:`id_to_token_map_py`, :attr:`token_to_id_map_py`), where
            :attr:`id_to_token_map` and :attr:`token_to_id_map` are
            TF `HashTable` instances, and :attr:`id_to_token_map_py` and
            :attr:`token_to_id_map_py` are python `defaultdict` instances.
        """
        with gfile.GFile(filename) as vocab_file:
            vocab = list(line.strip() for line in vocab_file)

        if self._bos_token in vocab:
            raise ValueError("Special token already exists in the "
                             "vocabulary %s" % self._bos_token)
        if self._eos_token in vocab:
            raise ValueError("Special token already exists in the "
                             "vocabulary %s" % self._eos_token)
        if self._unk_token in vocab:
            raise ValueError("Special token already exists in the "
                             "vocabulary %s" % self._unk_token)
        if self._padding_token in vocab:
            raise ValueError("Special padding token already exists in the "
                             "vocabulary %s, it is an empty token by default"
                             % self._padding_token)

        # Placing _padding_token at the beginning to make sure it take index 0.
        vocab = [self._padding_token, self._bos_token, self._eos_token,
                 self._unk_token] + vocab
        vocab_size = len(vocab)
        vocab_idx = np.arange(vocab_size)
        unk_token_idx = vocab_size - 1

        # Creates TF maps
        id_to_token_map = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(
                vocab_idx, vocab, key_dtype=tf.int64, value_dtype=tf.string),
            self._unk_token)

        token_to_id_map = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(
                vocab, vocab_idx, key_dtype=tf.string, value_dtype=tf.int64),
            unk_token_idx)

        # Creates python maps to interface with python code
        id_to_token_map_py = _make_defaultdict(
            vocab_idx, vocab, self._unk_token)
        token_to_id_map_py = _make_defaultdict(
            vocab, vocab_idx, unk_token_idx)

        return id_to_token_map, token_to_id_map, \
               id_to_token_map_py, token_to_id_map_py

    @property
    def id_to_token_map(self):
        """The `HashTable` instance that maps from token index to the
        string form.
        """
        return self._id_to_token_map

    @property
    def token_to_id_map(self):
        """The `HashTable` instance that maps from token string to the
        index.
        """
        return self._token_to_id_map

    @property
    def id_to_token_map_py(self):
        """The `defaultdict` instance that maps from token index to the
        string form.
        """
        return self._id_to_token_map_py

    @property
    def token_to_id_map_py(self):
        """The `defaultdict` instance that maps from token string to the
        index.
        """
        return self._token_to_id_map_py

    @property
    def vocab_size(self):
        """The vocabulary size.
        """
        return len(self.token_to_id_map_py)

    @property
    def bos_token(self):
        """A string of the special token indicating the beginning of sequence.
        """
        return self._bos_token

    @property
    def eos_token(self):
        """A string of the special token indicating the end of sequence.
        """
        return self._eos_token

    @property
    def unk_token(self):
        """A string of the special token indicating unkown token.
        """
        return self._unk_token

    @property
    def special_tokens(self):
        """The list of special tokens :attr:`[bos_token, eos_token, unk_token]`.
        """
        return [self._bos_token, self._eos_token, self._unk_token]
