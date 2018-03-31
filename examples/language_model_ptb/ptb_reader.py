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
"""Utilities for preprocessing and iterating over the PTB data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, too-many-locals

import os
import numpy as np

import tensorflow as tf

import texar as tx

def ptb_iterator(data, batch_size, num_steps):
    """Iterates through the ptb data.
    """
    data_length = len(data)
    batch_length = data_length // batch_size

    data = np.asarray(data[:batch_size*batch_length])
    data = data.reshape([batch_size, batch_length])

    epoch_size = (batch_length - 1) // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps : (i+1) * num_steps]
        y = data[:, i * num_steps + 1 : (i+1) * num_steps + 1]
        yield (x, y)

def prepare_data(data_path):
    """Preprocess PTB data.
    """
    train_path = os.path.join(data_path, "ptb.train.txt")
    if not tf.gfile.Exists(train_path):
        url = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'
        tx.data.maybe_download(url, data_path, extract=True)
        data_path = os.path.join(data_path, 'simple-examples', 'data')

    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = tx.data.make_vocab(
        train_path, newline_token="<EOS>", return_type="dict")
    assert len(word_to_id) == 10000

    train_text = tx.data.read_words(
        train_path, newline_token="<EOS>")
    train_text_id = [word_to_id[w] for w in train_text if w in word_to_id]

    valid_text = tx.data.read_words(
        valid_path, newline_token="<EOS>")
    valid_text_id = [word_to_id[w] for w in valid_text if w in word_to_id]

    test_text = tx.data.read_words(
        test_path, newline_token="<EOS>")
    test_text_id = [word_to_id[w] for w in test_text if w in word_to_id]

    data = {
        "train_text": train_text,
        "valid_text": valid_text,
        "test_text": test_text,
        "train_text_id": train_text_id,
        "valid_text_id": valid_text_id,
        "test_text_id": test_text_id,
        "vocab": word_to_id,
        "vocab_size": len(word_to_id)
    }
    return data
