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
"""Data read/write utilities for Transformer.
"""
import os
import codecs
import six
import numpy as np

# pylint: disable=no-member

def load_data_numpy(input_dir, prefix):
    train_data = np.load(os.path.join(input_dir,\
        prefix + 'train.npy')).tolist()
    dev_data = np.load(os.path.join(input_dir,\
        prefix + 'valid.npy')).tolist()
    test_data = np.load(os.path.join(input_dir,\
        prefix + 'test.npy')).tolist()
    print('train data size:{}'.format(len(train_data)))
    return train_data, dev_data, test_data

def seq2seq_pad_concat_convert(xy_batch, eos_id=2, bos_id=1):
    """
    Args:
        xy_batch (list of tuple of two numpy.ndarray-s or cupy.ndarray-s):
            xy_batch[i][0] is an array
            of token ids of i-th input sentence in a minibatch.
            xy_batch[i][1] is an array
            of token ids of i-th target sentence in a minibatch.
            The shape of each array is `(sentence length, )`.
        eos_id: The index of end-of-sentence special token in the
            dictionary.

    Returns:
        Tuple of Converted array.
            (input_sent_batch_array, target_sent_batch_input_array,
            target_sent_batch_output_array).
            The shape of each array is `(batchsize, max_sentence_length)`.
            All sentences are padded with 0 to reach max_sentence_length.
    """

    x_seqs, y_seqs = zip(*xy_batch)
    x_block = _concat_examples(x_seqs, padding=0)
    y_block = _concat_examples(y_seqs, padding=0)

    # Add EOS
    x_block = np.pad(x_block, ((0, 0), (0, 1)), 'constant',
                     constant_values=0)
    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id

    y_out_block = np.pad(y_block, ((0, 0), (0, 1)), 'constant',
                         constant_values=0)
    for i_batch, seq in enumerate(y_seqs):
        y_out_block[i_batch, len(seq)] = eos_id

    # Add BOS in target language
    y_in_block = np.pad(y_block, ((0, 0), (1, 0)), 'constant',
                        constant_values=bos_id)
    return x_block, y_in_block, y_out_block

def source_pad_concat_convert(x_seqs, eos_id=2, bos_id=1):
    """
    This function is used when testing the model without target input.
    """
    x_block = _concat_examples(x_seqs, padding=0)

    # add EOS
    x_block = np.pad(x_block, ((0, 0), (0, 1)), 'constant', constant_values=0)
    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id
    return x_block

def _concat_examples(arrays, padding=0):
    if len(arrays) == 0:
        raise ValueError('batch is empty')

    first_elem = arrays[0]
    assert isinstance(first_elem, np.ndarray)

    shape = np.array(arrays[0].shape, dtype=int)
    for array in arrays[1:]:
        if np.any(shape != array.shape):
            np.maximum(shape, array.shape, shape)
    shape = tuple(np.insert(shape, 0, len(arrays)))

    result = np.full(shape, padding, dtype=arrays[0].dtype)
    for i in six.moves.range(len(arrays)):
        src = arrays[i]
        slices = tuple(slice(dim) for dim in src.shape)
        result[(i,) + slices] = src
    return result

def write_words(words_list, filename):
    with codecs.open(filename, 'w+', 'utf-8') as myfile:
        for words in words_list:
            myfile.write(' '.join(words) + '\n')

