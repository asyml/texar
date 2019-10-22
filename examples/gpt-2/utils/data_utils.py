# Copyright 2019 The Texar Authors. All Rights Reserved.
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
Utils of data preprocessing for GPT2 training.
"""

import os
import collections
import tensorflow as tf

# pylint: disable=invalid-name, too-many-arguments


def process_single_text(raw_text, max_seq_length, encoder,
                        BOS_token, EOS_token, PAD_token):
    """Processes a single piece of text. Performs BPE encoding,
    converting to indexes, truncation, and padding, etc.
    """
    # BPE
    tokens = encoder.encode(raw_text)

    # Truncate
    max_len = max_seq_length
    if BOS_token is not None and len(BOS_token) > 0:
        max_len -= 1
    if EOS_token is not None and len(EOS_token) > 0:
        max_len -= 1
    tokens = tokens[:max_len]

    # Append special tokens
    if BOS_token is not None and len(BOS_token) > 0:
        tokens = [encoder.encoder[BOS_token]] + tokens
    if EOS_token is not None and len(EOS_token) > 0:
        tokens = tokens + [encoder.encoder[EOS_token]]

    token_length = len(tokens)

    # Pad
    PAD_token_id = encoder.encoder[PAD_token]
    while len(tokens) < max_seq_length:
        tokens.append(PAD_token_id)

    assert len(tokens) == max_seq_length

    return tokens, token_length


def read_raw_data(data_fn):
    """
    Reads raw data from a file. Each line contains one example.
    """
    examples = []
    with open(data_fn, "r") as fin:
        for line in fin:
            examples.append(line.strip())
    return examples


def file_based_convert_examples_to_features(
        examples, max_seq_length, encoder, output_file,
        BOS_token="<|endoftext|>", EOS_token="<|endoftext|>",
        PAD_token="<|endoftext|>"):
    """Converts a set of examples to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (_, example) in enumerate(examples):

        text_ids, length = process_single_text(
            example, max_seq_length, encoder, BOS_token, EOS_token, PAD_token)

        def _create_int_feature(values):
            return tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))

        features = collections.OrderedDict()
        features["text_ids"] = _create_int_feature(text_ids)
        features["length"] = _create_int_feature([length])

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def prepare_TFRecord_data(data_dir, max_seq_length, encoder, output_dir):
    """
    Args:
        data_dir: The input data directory.
        max_seq_length: Max sequence length.
        output_dir: The directory to save the TFRecord files in.
    """
    train_fn = os.path.join(data_dir, "train.txt")
    if os.path.isfile(train_fn):
        tf.logging.info("Processing %s" % train_fn)
        train_examples = read_raw_data(train_fn)
        train_file = os.path.join(output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, max_seq_length, encoder, train_file)

    dev_fn = os.path.join(data_dir, "dev.txt")
    if os.path.isfile(dev_fn):
        tf.logging.info("Processing %s" % dev_fn)
        eval_examples = read_raw_data(dev_fn)
        eval_file = os.path.join(output_dir, "dev.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, max_seq_length, encoder, eval_file)

    test_fn = os.path.join(data_dir, "test.txt")
    if os.path.isfile(test_fn):
        tf.logging.info("Processing %s" % test_fn)
        test_examples = read_raw_data(test_fn)
        test_file = os.path.join(output_dir, "test.tf_record")
        file_based_convert_examples_to_features(
            test_examples, max_seq_length, encoder, test_file, EOS_token=None)
