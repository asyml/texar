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

def _make_defaultdict(keys, values, default_value):
    """Creates a python defaultdict.

    Args:
        keys: A list of keys.
        values: A list of values correspond to keys. The two lists `keys` and
            `values` must be of the same length.
        default_value: default value returned when key is missing.

    Returns:
        A python defaultdict instance.
    """
    dict_ = defaultdict(lambda: default_value)
    for k, v in zip(keys, values):  # pylint: disable=invalid-name
        dict_[k] = v


    return dict_

class Vocab(object):    # pylint: disable=too-many-instance-attributes
    """Vocabulary class that loads vocabulary from file, and maintains mapping
    tables between token strings and indexes.
    """

    def __init__(self,
                 filename,
                 bos_token="<BOS>",
                 eos_token="<EOS>",
                 unk_token="<UNK>"):
        """Creates the vocabulary.

        Args:
            filename: Path to the vocabulary file where each line contains one
                word. Each word is indexed with its line number (starting from
                0).
            bos_token: A special token that will be added to the beginning of
                sequences.
            eos_token: A special token that will be added to the end of
                sequences.
            unk_token: A special token that will replace all unkown tokens.
        """
        self._filename = filename
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._unk_token = unk_token

        self._id_to_token_map, self._token_to_id_map, \
            self._id_to_token_map_py, self._token_to_id_map_py = \
            self.load(self._filename)

    def load(self, filename):
        """Loads the vocabulary from the file.

        Args:
            filename: Path to the vocabulary file.

        Returns:
            A tuple of TF and python Mapping tables between word string and
            index, (id_to_token_map, token_to_id_map, id_to_token_map_py,
            token_to_id_map_py), where id_to_token_map and token_to_id_map are
            TF HashTable instances, and id_to_token_map_py and
            token_to_id_map_py are python defaultdict instances.
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

        vocab += [self._bos_token, self._eos_token, self._unk_token]
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

        # Creates python built-in maps to interface with python code
        id_to_token_map_py = _make_defaultdict(
            vocab_idx, vocab, self._unk_token)
        token_to_id_map_py = _make_defaultdict(
            vocab, vocab_idx, unk_token_idx)

        return id_to_token_map, token_to_id_map, \
               id_to_token_map_py, token_to_id_map_py

    @property
    def id_to_token_map(self):
        """Returns the HashTable instance that maps from token index to the
        string form.
        """
        return self._id_to_token_map

    @property
    def token_to_id_map(self):
        """Returns the HashTable instance that maps from token string to the
        index.
        """
        return self._token_to_id_map

    @property
    def id_to_token_map_py(self):
        """Returns the defaultdict instance that maps from token index to the
        string form.
        """
        return self._id_to_token_map_py

    @property
    def token_to_id_map_py(self):
        """Returns the defaultdict instance that maps from token string to the
        index.
        """
        return self._token_to_id_map_py

    @property
    def vocab_size(self):
        """Returns the vocab size.

        Returns:
            An integer.
        """
        return len(self.token_to_id_map_py)

    @property
    def bos_token(self):
        """Returns the special token indicating the beginning of sequence.
        """
        return self._bos_token

    @property
    def eos_token(self):
        """Returns the special token indicating the end of sequence.
        """
        return self._eos_token

    @property
    def unk_token(self):
        """Returns the special token indicating unkown token.
        """
        return self._unk_token

    @property
    def special_tokens(self):
        """Returns the list of special tokens.
        """
        return [self._bos_token, self._eos_token, self._unk_token]
