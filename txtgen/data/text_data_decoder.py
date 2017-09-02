#
"""Helper functions and classes for decoding text data which are used after
reading raw text data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import data_decoder


class TextDataDecoder(data_decoder.DataDecoder):   # pylint: disable=too-many-instance-attributes
    """A text data decoder that decodes raw text data, including splitting on
    word or character level, truncation, inserting special tokens, mapping
    text units to indexes, etc.
    """

    def __init__(self,  # pylint: disable=too-many-arguments
                 split_level="word",
                 delimiter=" ",
                 bos_token=None,
                 eos_token=None,
                 max_seq_length=None,
                 token_to_id_map=None,
                 text_tensor_name="text",
                 length_tensor_name="length",
                 text_id_tensor_name="text_ids"):
        """Constructs the decoder.

        Args:
            split_level: The name of split level on which text sequence is
                split. Either "word" or "character".
            delimiter: The delimiter character used when splitting on word
                level.
            bos_token: (optional) Special token added to the beginning of
                sequences. If it is `None` (default) or an empty string, no
                BOS token is added.
            eos_token: (optional) Special tokan added to the end of sequences.
                If it is `None` (default) or an empty string, no EOS token is
                added.
            max_seq_length: (optional) Maximum length of output sequences.
                Tokens exceed the maximum length will be truncated. The length
                does not include any added bos_token and eos_token. If not
                given, no truncation is performed.
            token_to_id_map: (optional) A HashTable instance that maps token
                strings to integer indexes. If not given, the decoder will not
                decode text into indexes. `bos_token` and `eos_token` (if given)
                should have entries in the `token_to_id_map` (if given).
            text_tensor_name: Name of the text tensor results. Used as a key to
                retrieve the text tensor.
            length_tensor_name: Name of the text length tensor results.
            text_id_tensor_name: Name of the text index tensor results.
        """
        self._split_level = split_level
        self._delimiter = delimiter
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._max_seq_length = max_seq_length
        self._token_to_id_map = token_to_id_map
        self._text_tensor_name = text_tensor_name
        self._text_id_tensor_name = text_id_tensor_name
        self._length_tensor_name = length_tensor_name

    def decode(self, data, items):
        """Decodes the data to return the tensors specified by the list of
        items.

        Args:
            data: The text data to decode.
            items: A list of strings, each of which is the name of the resulting
                tensors to retrieve.

        Returns:
            A list of tensors, each of which corresponds to each item. If
            `token_to_id_map` is not given when constructing the decoder,
            returns `None` for the token index item.
        """
        # Split
        if self._split_level == "word":
            tokens = tf.string_split([data], delimiter=self._delimiter).values
        elif self._split_level == "character":
            raise NotImplementedError
        else:
            raise ValueError("Unknown split level: %s" % self._split_level)

        # Truncate
        if self._max_seq_length is not None:
            tokens = tokens[:self._max_seq_length]

        # Add BOS/EOS tokens
        if self._bos_token is not None:
            tokens = tf.concat([[self._bos_token], tokens], axis=0)
        if self._eos_token is not None:
            tokens = tf.concat([tokens, [self._eos_token]], axis=0)

        # Map to index
        token_ids = None
        if self._token_to_id_map is not None:
            token_ids = self._token_to_id_map.lookup(tokens)

        outputs = {
            self._text_tensor_name: tokens,
            self._length_tensor_name: tf.size(tokens),
            self._text_id_tensor_name: token_ids
        }
        return [outputs[item] for item in items]

    def list_items(self):
        """Returns the list of item names that the decoder can produce.

        Returns:
            A list of strings can be passed to `decode()`.
        """
        return [self._text_tensor_name,
                self._length_tensor_name,
                self._text_id_tensor_name]

    @property
    def text_tensor_name(self):
        """Returns the name of text tensor
        """
        return self._text_tensor_name

    @property
    def length_tensor_name(self):
        """Returns the name of length tensor
        """
        return self._length_tensor_name

    @property
    def text_id_tensor_name(self):
        """Returns the name of text index tensor
        """
        return self._text_id_tensor_name
