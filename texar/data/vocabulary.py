# -*- coding: utf-8 -*-
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
Helper functions and classes for vocabulary processing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
from collections import defaultdict

import tensorflow as tf
from tensorflow import gfile
import numpy as np

from texar.utils.utils import dict_lookup

# pylint: disable=too-few-public-methods, invalid-name
# pylint: disable=too-many-instance-attributes, too-many-arguments

__all__ = [
    "SpecialTokens",
    "Vocab"
]

class SpecialTokens(object):
    """Special tokens, including :attr:`PAD`, :attr:`BOS`, :attr:`EOS`,
    :attr:`UNK`. These tokens will by default have token ids 0, 1, 2, 3,
    respectively.
    """
    PAD = "<PAD>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"


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
    for k, v in zip(keys, values):
        dict_[k] = v

    return dict_


class Vocab(object):
    """Vocabulary class that loads vocabulary from file, and maintains mapping
    tables between token strings and indexes.

    Each line of the vocab file should contains one vocabulary token, e.g.,::

        vocab_token_1
        vocab token 2
        vocab       token | 3 .
        ...

    Args:
        filename (str): Path to the vocabulary file where each line contains
            one token.
        bos_token (str): A special token that will be added to the beginning of
            sequences.
        eos_token (str): A special token that will be added to the end of
            sequences.
        unk_token (str): A special token that will replace all unknown tokens
            (tokens not included in the vocabulary).
        pad_token (str): A special token that is used to do padding.
    """

    def __init__(self,
                 filename,
                 pad_token=SpecialTokens.PAD,
                 bos_token=SpecialTokens.BOS,
                 eos_token=SpecialTokens.EOS,
                 unk_token=SpecialTokens.UNK):
        self._filename = filename
        self._pad_token = pad_token
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._unk_token = unk_token

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
            TF :tf_main:`HashTable <contrib/lookup/HashTable>` instances,
            and :attr:`id_to_token_map_py` and
            :attr:`token_to_id_map_py` are python `defaultdict` instances.
        """
        with gfile.GFile(filename) as vocab_file:
            # Converts to 'unicode' (Python 2) or 'str' (Python 3)
            vocab = list(tf.compat.as_text(line.strip()) for line in vocab_file)

        warnings.simplefilter("ignore", UnicodeWarning)

        if self._bos_token in vocab:
            raise ValueError("Special begin-of-seq token already exists in the "
                             "vocabulary: '%s'" % self._bos_token)
        if self._eos_token in vocab:
            raise ValueError("Special end-of-seq token already exists in the "
                             "vocabulary: '%s'" % self._eos_token)
        if self._unk_token in vocab:
            raise ValueError("Special UNK token already exists in the "
                             "vocabulary: '%s'" % self._unk_token)
        if self._pad_token in vocab:
            raise ValueError("Special padding token already exists in the "
                             "vocabulary: '%s'" % self._pad_token)

        warnings.simplefilter("default", UnicodeWarning)

        # Places _pad_token at the beginning to make sure it take index 0.
        vocab = [self._pad_token, self._bos_token, self._eos_token,
                 self._unk_token] + vocab
        # Must make sure this is consistent with the above line
        unk_token_idx = 3
        vocab_size = len(vocab)
        vocab_idx = np.arange(vocab_size)

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

    def map_ids_to_tokens(self, ids):
        """Maps ids into text tokens.

        The returned tokens are a Tensor.

        Args:
            ids: An `int` tensor of token ids.

        Returns:
            A tensor of text tokens of the same shape.
        """
        return self.id_to_token_map.lookup(tf.to_int64(ids))

    def map_tokens_to_ids(self, tokens):
        """Maps text tokens into ids.

        The returned ids are a Tensor.

        Args:
            tokens: An tensor of text tokens.

        Returns:
            A tensor of token ids of the same shape.
        """
        return self.token_to_id_map.lookup(tokens)

    def map_ids_to_tokens_py(self, ids):
        """Maps ids into text tokens.

        The input :attr:`ids` and returned tokens are both python
        arrays or list.

        Args:
            ids: An `int` numpy arry or (possibly nested) list of token ids.

        Returns:
            A numpy array of text tokens of the same shape as :attr:`ids`.
        """
        return dict_lookup(self.id_to_token_map_py, ids, self.unk_token)

    def map_tokens_to_ids_py(self, tokens):
        """Maps text tokens into ids.

        The input :attr:`tokens` and returned ids are both python
        arrays or list.

        Args:
            tokens: A numpy array or (possibly nested) list of text tokens.

        Returns:
            A numpy array of token ids of the same shape as :attr:`tokens`.
        """
        return dict_lookup(self.token_to_id_map_py, tokens, self.unk_token_id)

    @property
    def id_to_token_map(self):
        """The :tf_main:`HashTable <contrib/lookup/HashTable>`instance that
        maps from token index to the string form.
        """
        return self._id_to_token_map

    @property
    def token_to_id_map(self):
        """The :tf_main:`HashTable <contrib/lookup/HashTable>` instance
        that maps from token string to the index.
        """
        return self._token_to_id_map

    @property
    def id_to_token_map_py(self):
        """The python `defaultdict` instance that maps from token index to the
        string form.
        """
        return self._id_to_token_map_py

    @property
    def token_to_id_map_py(self):
        """The python `defaultdict` instance that maps from token string to the
        index.
        """
        return self._token_to_id_map_py

    @property
    def size(self):
        """The vocabulary size.
        """
        return len(self.token_to_id_map_py)

    @property
    def bos_token(self):
        """A string of the special token indicating the beginning of sequence.
        """
        return self._bos_token

    @property
    def bos_token_id(self):
        """The `int` index of the special token indicating the beginning
        of sequence.
        """
        return self.token_to_id_map_py[self._bos_token]

    @property
    def eos_token(self):
        """A string of the special token indicating the end of sequence.
        """
        return self._eos_token

    @property
    def eos_token_id(self):
        """The `int` index of the special token indicating the end
        of sequence.
        """
        return self.token_to_id_map_py[self._eos_token]

    @property
    def unk_token(self):
        """A string of the special token indicating unknown token.
        """
        return self._unk_token

    @property
    def unk_token_id(self):
        """The `int` index of the special token indicating unknown token.
        """
        return self.token_to_id_map_py[self._unk_token]

    @property
    def pad_token(self):
        """A string of the special token indicating padding token. The
        default padding token is an empty string.
        """
        return self._pad_token

    @property
    def pad_token_id(self):
        """The `int` index of the special token indicating padding token.
        """
        return self.token_to_id_map_py[self._pad_token]

    @property
    def special_tokens(self):
        """The list of special tokens
        [:attr:`pad_token`, :attr:`bos_token`, :attr:`eos_token`,
        :attr:`unk_token`].
        """
        return [self._pad_token, self._bos_token, self._eos_token,
                self._unk_token]
