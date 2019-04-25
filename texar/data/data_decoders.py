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
Helper functions and classes for decoding text data which are used after
reading raw text data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import data_decoder

from texar.data.vocabulary import SpecialTokens
from texar.utils import dtypes
from texar.hyperparams import HParams


# pylint: disable=too-many-instance-attributes, too-many-arguments,
# pylint: disable=no-member, invalid-name

__all__ = [
    "ScalarDataDecoder",
    "TextDataDecoder",
    "VarUttTextDataDecoder",
    "TFRecordDataDecoder",
]

def _append_token(token):
    return token is not None and token != ""

class ScalarDataDecoder(data_decoder.DataDecoder):
    """A data decoder that decodes a scalar, e.g., int label or float number.

    The only operation is to cast the data into a specified data type.

    Args:
        dtype: A :tf_main:`tf DType <DType>` that data is cast into. Can be
            `tf.int32` or `tf.float32`.
        data_name (str): Name of the decoded data.
    """

    def __init__(self, dtype=tf.int32, data_name="data"):
        self._dtype = dtype
        self._data_name = data_name
        if self._data_name is None:
            self._data_name = "data"

    def __call__(self, data):
        outputs = self.decode(data, self.list_items())
        return dict(zip(self.list_items(), outputs))

    def decode(self, data, items):
        """Decodes the data to return the tensors specified by the list of
        items.

        Args:
            data: The scalar data to decode.
            items: A list of strings, each of which is the name of the resulting
                tensors to retrieve.

        Returns:
            A list of tensors, each of which corresponds to each item.
        """
        data = tf.reshape(data, shape=[])
        if data.dtype is tf.string:
            decoded_data = tf.string_to_number(data, out_type=self._dtype)
        else:
            decoded_data = tf.cast(data, self._dtype)
        outputs = {
            self._data_name: decoded_data
        }
        return [outputs[item] for item in items]

    def list_items(self):
        """Returns the list of item names that the decoder can produce.

        Returns:
            A list of strings can be passed to :meth:`decode()`.
        """
        return [self._data_name]

    @property
    def data_tensor_name(self):
        """The name of the data tensor.
        """
        return self._data_name

class TextDataDecoder(data_decoder.DataDecoder):
    """A text data decoder that decodes raw text data.

    Operations include splitting on word or character level, truncation,
    inserting special tokens, mapping text units to indexes, etc.

    Args:
        split_level (str): The name of split level on which text sequence is
            split. Either "word" or "char".
        delimiter (str): The delimiter character used when splitting on word
            level.
        bos_token (str, optional): Special token added to the beginning of
            sequences. If it is `None` (default) or an empty string, no
            BOS token is added.
        eos_token (str, optional): Special tokan added to the end of
            sequences. If it is `None` (default) or an empty string, no EOS
            token is added.
        max_seq_length (int, optional): Maximum length of output sequences.
            Tokens exceeding the maximum length will be truncated. The length
            does not include any added bos_token and eos_token. If not
            given, no truncation is performed.
        token_to_id_map (optional): A
            :class:`~tensorflow.contrib.lookup.HashTable` instance that maps
            token strings to integer indexes. If not given, the decoder will
            not decode text into indexes. :attr:`bos_token` and
            :attr:`eos_token` (if given) should have entries in the
            :attr:`token_to_id_map` (if given).
        text_tensor_name (str): Name of the text tensor results. Used as a
            key to retrieve the text tensor.
        length_tensor_name (str): Name of the text length tensor results.
        text_id_tensor_name (str): Name of the text index tensor results.
    """

    def __init__(self,
                 split_level="word",
                 delimiter=" ",
                 bos_token=None,
                 eos_token=None,
                 max_seq_length=None,
                 token_to_id_map=None,
                 text_tensor_name="text",
                 length_tensor_name="length",
                 text_id_tensor_name="text_ids"):
        self._split_level = split_level
        self._delimiter = delimiter
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._max_seq_length = max_seq_length
        self._token_to_id_map = token_to_id_map
        self._text_tensor_name = text_tensor_name
        self._text_id_tensor_name = text_id_tensor_name
        self._length_tensor_name = length_tensor_name
        self._added_length = 0

    def __call__(self, data):
        outputs = self.decode(data, self.list_items())
        return dict(zip(self.list_items(), outputs))

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
        elif self._split_level == "char":
            raise NotImplementedError
        else:
            raise ValueError("Unknown split level: %s" % self._split_level)

        # Truncate
        if self._max_seq_length is not None:
            tokens = tokens[:self._max_seq_length]

        # Add BOS/EOS tokens
        if _append_token(self._bos_token):
            tokens = tf.concat([[self._bos_token], tokens], axis=0)
            self._added_length += 1
        if _append_token(self._eos_token):
            tokens = tf.concat([tokens, [self._eos_token]], axis=0)
            self._added_length += 1

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
            A list of strings can be passed to :meth:`decode()`.
        """
        return [self._text_tensor_name,
                self._length_tensor_name,
                self._text_id_tensor_name]

    @property
    def text_tensor_name(self):
        """The name of text tensor.
        """
        return self._text_tensor_name

    @text_tensor_name.setter
    def text_tensor_name(self, name):
        self._text_tensor_name = name

    @property
    def length_tensor_name(self):
        """The name of length tensor.
        """
        return self._length_tensor_name

    @length_tensor_name.setter
    def length_tensor_name(self, name):
        self._length_tensor_name = name

    @property
    def text_id_tensor_name(self):
        """The name of text index tensor.
        """
        return self._text_id_tensor_name

    @text_id_tensor_name.setter
    def text_id_tensor_name(self, name):
        self._text_id_tensor_name = name

    @property
    def added_length(self):
        """The added text length due to appended bos and eos tokens.
        """
        return self._added_length

class VarUttTextDataDecoder(data_decoder.DataDecoder):
    """A text data decoder that decodes raw text data. Each data is considered
    to be multiple sentences concatenated by a delimiter.

    Operations include splitting on word or character level, truncation,
    inserting special tokens, mapping text units to indexes, etc.

    Args:
        split_level (str): The name of split level on which text sequence is
            split. Either "word" or "char".
        delimiter (str): The delimiter character used when splitting on word
            level.
        bos_token (str, optional): Special token added to the beginning of
            sequences. If it is `None` (default) or an empty string, no
            BOS token is added.
        eos_token (str, optional): Special tokan added to the end of
            sequences. If it is `None` (default) or an empty string, no EOS
            token is added.
        max_seq_length (int): Maximum length of each sequence.
            Tokens exceed the maximum length will be truncated. Additional
            padding will be done to ensure output sequence all reach this
            number. The length does not include any added bos_token and eos_
            token.
        max_utterance_cnt (int): Maximum number of sequences.
            Additional empty sentences will be added to
            ensure the respective dimension of the output tensor has size
            :attr:`max_utterance_cnt`. The output item named by
            :meth:`utterance_cnt_tensor_name` contains the actual number of
            utterance in the data.
        token_to_id_map (optional): A
            :class:`~tensorflow.contrib.lookup.HashTable` instance that maps
            token strings to integer indexes. If not given, the decoder will
            not decode text into indexes. :attr:`bos_token` and
            :attr:`eos_token` (if given) should have entries in the
            :attr:`token_to_id_map` (if given).
        text_tensor_name (str): Name of the text tensor results. Used as a
            key to retrieve the text tensor.
        length_tensor_name (str): Name of the text length tensor results.
        text_id_tensor_name (str): Name of the text index tensor results.
    """

    def __init__(self,
                 split_level="word",
                 delimiter=" ",
                 sentence_delimiter="|||",
                 bos_token=None,
                 eos_token=None,
                 max_seq_length=None,
                 max_utterance_cnt=None,
                 token_to_id_map=None,
                 text_tensor_name="text",
                 length_tensor_name="length",
                 text_id_tensor_name="text_ids",
                 utterance_cnt_tensor_name="utterance_cnt"):
        self._split_level = split_level
        self._delimiter = delimiter
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._max_seq_length = max_seq_length
        self._token_to_id_map = token_to_id_map
        self._text_tensor_name = text_tensor_name
        self._text_id_tensor_name = text_id_tensor_name
        self._length_tensor_name = length_tensor_name
        self._utterance_cnt_tensor_name = utterance_cnt_tensor_name
        self._sentence_delimiter = sentence_delimiter
        self._max_utterance_cnt = max_utterance_cnt
        self._added_length = 0

    def __call__(self, data):
        outputs = self.decode(data, self.list_items())
        return dict(zip(self.list_items(), outputs))

    def decode(self, data, items): # pylint: disable=too-many-locals
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

        sentences = tf.string_split([data],
                                    delimiter=self._sentence_delimiter).values

        # Truncate utterances
        if self._max_utterance_cnt:
            sentences = sentences[:self._max_utterance_cnt]
        utterance_cnt = tf.shape(sentences)[0]

        # Get (max) sentence length
        def _get_sent_length(s):
            raw_length = tf.size(
                tf.string_split([s], delimiter=self._delimiter).values)
            if self._max_seq_length:
                return tf.minimum(raw_length, self._max_seq_length)
            else:
                return raw_length

        raw_sent_length = tf.map_fn(
            _get_sent_length, sentences, dtype=tf.int32)
        sent_length = self._max_seq_length
        if not sent_length:
            sent_length = tf.reduce_max(raw_sent_length)
        if _append_token(self._eos_token):
            raw_sent_length += 1
            sent_length += 1
            self._added_length += 1
        if _append_token(self._bos_token):
            raw_sent_length += 1
            sent_length += 1
            self._added_length += 1

        def _trunc_and_pad(s, pad_token, max_length):
            if self._max_seq_length:
                s = s[:self._max_seq_length]
            if _append_token(self._bos_token):
                s = np.append([self._bos_token], s)
            if _append_token(self._eos_token):
                s = np.append(s, [self._eos_token])
            s = np.append(s, [pad_token]*(max_length-s.size))
            return s

        # Split each sentence to tokens, and pad them to a same length.
        # This is necessary to treat all sentences as a single tensor.
        split_sentences = tf.map_fn(
            lambda s: tf.py_func(
                _trunc_and_pad,
                [
                    tf.string_split([s], delimiter=self._delimiter).values,
                    SpecialTokens.PAD,
                    sent_length
                ],
                tf.string),
            sentences, dtype=tf.string
        )

        split_sentences = tf.reshape(split_sentences,
                                     [utterance_cnt, sent_length])

        # Map to index
        token_ids = None
        if self._token_to_id_map is not None:
            token_ids = self._token_to_id_map.lookup(split_sentences)

        outputs = {
            self._text_tensor_name: split_sentences,
            self._length_tensor_name: raw_sent_length,
            self._utterance_cnt_tensor_name: tf.shape(sentences)[0],
            self._text_id_tensor_name: token_ids
        }
        return [outputs[item] for item in items]

    def list_items(self):
        """Returns the list of item names that the decoder can produce.

        Returns:
            A list of strings can be passed to :meth:`decode()`.
        """
        return [
            self._text_tensor_name,
            self._length_tensor_name,
            self._text_id_tensor_name,
            self._utterance_cnt_tensor_name
        ]

    @property
    def text_tensor_name(self):
        """The name of text tensor.
        """
        return self._text_tensor_name

    @text_tensor_name.setter
    def text_tensor_name(self, name):
        self._text_tensor_name = name

    @property
    def utterance_cnt_tensor_name(self):
        """The name of the utterance count tensor.
        """
        return self._utterance_cnt_tensor_name

    @property
    def length_tensor_name(self):
        """The name of length tensor.
        """
        return self._length_tensor_name

    @length_tensor_name.setter
    def length_tensor_name(self, name):
        self._length_tensor_name = name

    @property
    def text_id_tensor_name(self):
        """The name of text index tensor.
        """
        return self._text_id_tensor_name

    @text_id_tensor_name.setter
    def text_id_tensor_name(self, name):
        self._text_id_tensor_name = name

    @property
    def added_length(self):
        """The added text length due to appended bos and eos tokens.
        """
        return self._added_length

class TFRecordDataDecoder(data_decoder.DataDecoder):
    """A data decoder that decodes a TFRecord file, e.g., the
    TFRecord file.

    The only operation is to parse the TFRecord data into a
    specified data type that can be accessed by features.

    Args:
        "feature_original_types" (dict): The feature names (str) with their
            data types and length types, key and value in pair
            `<feature_name: [dtype, feature_len_type, len]>`, type of
            `dtype` can be `tf DType <DType>` or `str`, e.g., 'tf.int32',
            'tf.float32', etc.

            - `feature_len_type` is of type `str` and can be \
            'FixedLenFeature' or 'VarLenFeature' for fixed length \
            features and non-fixed length features respectively.

            - `len` is optional, it is the length for the \
                'FixedLenFeature', can be a `int`.

        "feature_convert_types" (dict, optional): The feature names (str)
            with data types they are converted into, key and value in pair
            `<feature_name: dtype>`, `dtype` can be a `tf DType <DType>` or
            `str`, e.g., 'tf.int32', 'tf.float32', etc. If not set, data type
            conversion will not be performed.

            Be noticed that this converting process is after all the data
            are restored, `feature_original_types` has to be set firstly.

        "image_options" (dict, optional): Specifies the image feature name
            and performs image resizing, includes three fields:

            - "image_feature_name":
                A `str`, the name of the feature which contains
                the image data. If set, the image data
                will be restored in format `numpy.ndarray`.
            - "resize_height":
                A `int`, the height of the image after resizing.
            - "resize_width":
                A `int`, the width of the image after resizing

            If either `resize_height` or `resize_width` is not set,
            image data will be restored with original shape.
    """

    def __init__(self,
                 feature_original_types,
                 feature_convert_types,
                 image_options):
        self._feature_original_types = feature_original_types
        self._feature_convert_types = feature_convert_types
        self._image_options = image_options

    def __call__(self, data):
        outputs = self.decode(data, self.list_items())
        return dict(zip(self.list_items(), outputs))

    def _decode_image_str_byte(self,
                               image_option_feature,
                               decoded_data):
        # pylint: disable=no-self-use
        # Get image
        output_type = tf.uint8
        image_key = image_option_feature.get('image_feature_name')
        resize_height = image_option_feature.get("resize_height")
        resize_width = image_option_feature.get("resize_width")
        resize_method = image_option_feature.get("resize_method")
        if image_key is None:
            return
        image_byte = decoded_data.get(image_key)
        if image_byte is None:
            return

        def _find_resize_method(resize_method):
            if resize_method in {"AREA",
                                 "ResizeMethod.AREA",
                                 "tf.image.ResizeMethod.AREA",
                                 tf.image.ResizeMethod.AREA}:
                resize_method = tf.image.ResizeMethod.AREA
            elif resize_method in {"BICUBIC",
                                   "ResizeMethod.BICUBIC",
                                   "tf.image.ResizeMethod.BICUBIC",
                                   tf.image.ResizeMethod.BICUBIC}:
                resize_method = tf.image.ResizeMethod.BICUBIC
            elif resize_method in {"NEAREST_NEIGHBOR",
                                   "ResizeMethod.NEAREST_NEIGHBOR",
                                   "tf.image.ResizeMethod.NEAREST_NEIGHBOR",
                                   tf.image.ResizeMethod.NEAREST_NEIGHBOR}:
                resize_method = tf.image.ResizeMethod.AREA
            else:
                resize_method = tf.image.ResizeMethod.BILINEAR
            return resize_method

        # Decode image
        image_decoded = tf.cond(
            tf.image.is_jpeg(image_byte),
            lambda: tf.image.decode_jpeg(image_byte),
            lambda: tf.image.decode_png(image_byte))
        decoded_data[image_key] = image_decoded

        # Resize the image
        if resize_height and resize_width:
            resize_method = _find_resize_method(resize_method)
            image_resized = tf.image.resize_images(
                image_decoded,
                (resize_height, resize_width),
                method=resize_method)
            image_resized_converted = tf.cast(image_resized, output_type)
            decoded_data[image_key] = image_resized_converted
        return

    def decode(self, data, items):
        """Decodes the data to return the tensors specified by the list of
        items.

        Args:
            data: The TFRecord data(serialized example) to decode.
            items: A list of strings, each of which is the name of the resulting
                tensors to retrieve.

        Returns:
            A list of tensors, each of which corresponds to each item.
        """
        # pylint: disable=too-many-branches
        feature_description = dict()
        for key, value in  self._feature_original_types.items():
            shape = []
            if len(value) == 3:
                if isinstance(value[-1], int):
                    shape = [value[-1]]
                elif isinstance(value[-1], list):
                    shape = value
            if len(value) < 2 or value[1] == 'FixedLenFeature':
                feature_description.update(
                    {key: tf.FixedLenFeature(
                        shape,
                        dtypes.get_tf_dtype(value[0]))})
            elif value[1] == 'VarLenFeature':
                feature_description.update(
                    {key: tf.VarLenFeature(
                        dtypes.get_tf_dtype(value[0]))})
        decoded_data = tf.parse_single_example(data, feature_description)

        # Handle TFRecord containing images
        if isinstance(self._image_options, dict):
            self._decode_image_str_byte(
                self._image_options,
                decoded_data)
        elif isinstance(self._image_options, HParams):
            self._decode_image_str_byte(
                self._image_options.todict(),
                decoded_data)
        elif isinstance(self._image_options, list):
            _ = list(map(
                lambda x: self._decode_image_str_byte(x, decoded_data),
                self._image_options))

        # Convert Dtypes
        for key, value in self._feature_convert_types.items():
            from_type = decoded_data[key].dtype
            to_type = dtypes.get_tf_dtype(value)
            if from_type is to_type:
                continue
            elif to_type is tf.string:
                decoded_data[key] = tf.dtypes.as_string(decoded_data[key])
            elif from_type is tf.string:
                decoded_data[key] = tf.string_to_number(
                    decoded_data[key], to_type)
            else:
                decoded_data[key] = tf.cast(
                    decoded_data[key], to_type)
        outputs = decoded_data
        return [outputs[item] for item in items]

    def list_items(self):
        """Returns the list of item names that the decoder can produce.

        Returns:
            A list of strings can be passed to :meth:`decode()`.
        """
        return sorted(list(self._feature_original_types.keys()))
