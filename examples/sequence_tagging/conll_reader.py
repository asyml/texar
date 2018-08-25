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
"""Utilities for preprocessing and iterating over the CoNLL 2003 data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from collections import defaultdict
import numpy as np
import tensorflow as tf


# pylint: disable=invalid-name, too-many-locals

MAX_CHAR_LENGTH = 45
NUM_CHAR_PAD = 2

UNK_WORD, UNK_CHAR, UNK_NER = 0, 0, 0
PAD_WORD, PAD_CHAR, PAD_NER = 1, 1, 1

# Regular expressions used to normalize digits.
DIGIT_RE = re.compile(r"\d")


def create_vocabs(train_path, dev_path, test_path, normalize_digits=True, min_occur=1, glove_dict=None):
    word_vocab = defaultdict(lambda: len(word_vocab))
    word_count = defaultdict(lambda: 0)
    char_vocab = defaultdict(lambda: len(char_vocab))
    ner_vocab = defaultdict(lambda: len(ner_vocab))

    UNK_WORD = word_vocab["<unk>"]
    PAD_WORD = word_vocab["<pad>"]
    UNK_CHAR = char_vocab["<unk>"]
    PAD_CHAR = char_vocab["<pad>"]
    UNK_NER = ner_vocab["<unk>"]
    PAD_NER = ner_vocab["<pad>"]

    print("Creating Vocabularies:")

    for file_path in [train_path, dev_path, test_path]:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split(' ')
                for char in tokens[1]:
                    cid = char_vocab[char]

                word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
                ner = tokens[4]

                if glove_dict is not None and (word in glove_dict or word.lower() in glove_dict):
                    word_count[word] += min_occur + 1
                elif file_path == train_path:
                    word_count[word] += 1

                nid = ner_vocab[ner]

    print("Total Vocabulary Size: %d" % len(word_count))
    for word in word_count:
        if word_count[word] > min_occur:
            wid = word_vocab[word]

    print("Word Vocabulary Size: %d" % len(word_vocab))
    print("Character Alphabet Size: %d" % len(char_vocab))
    print("NER Alphabet Size: %d" % len(ner_vocab))

    word_vocab = defaultdict(lambda: UNK_WORD, word_vocab)
    char_vocab = defaultdict(lambda: UNK_CHAR, char_vocab)
    ner_vocab = defaultdict(lambda: UNK_NER, ner_vocab)

    i2w = {v: k for k, v in word_vocab.items()}
    i2n = {v: k for k, v in ner_vocab.items()}
    return (word_vocab, char_vocab, ner_vocab), (i2w, i2n)


def read_data(source_path, word_vocab, char_vocab, ner_vocab, normalize_digits=True):
    data = []
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLReader(source_path, word_vocab, char_vocab, ner_vocab)
    inst = reader.getNext(normalize_digits)
    while inst is not None:
        counter += 1
        sent = inst.sentence
        data.append([sent.word_ids, sent.char_id_seqs, inst.ner_ids])
        inst = reader.getNext(normalize_digits)

    reader.close()
    print("Total number of data: %d" % counter)
    return data


def iterate_batch(data, batch_size, shuffle=False):
    if shuffle:
        np.random.shuffle(data)

    for start_idx in range(0, len(data), batch_size):
        excerpt = slice(start_idx, start_idx + batch_size)
        batch = data[excerpt]

        batch_length = max([len(batch[i][0]) for i in range(len(batch))])

        wid_inputs = np.empty([len(batch), batch_length], dtype=np.int64)
        cid_inputs = np.empty([len(batch), batch_length, MAX_CHAR_LENGTH], dtype=np.int64)
        nid_inputs = np.empty([len(batch), batch_length], dtype=np.int64)
        masks = np.zeros([len(batch), batch_length], dtype=np.float32)
        lengths = np.empty(len(batch), dtype=np.int64)

        for i, inst in enumerate(batch):
            wids, cid_seqs, nids = inst

            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_CHAR
            cid_inputs[i, inst_size:, :] = PAD_CHAR
            nid_inputs[i, :inst_size] = nids
            nid_inputs[i, inst_size:] = PAD_NER
            masks[i, :inst_size] = 1.0

        yield wid_inputs, cid_inputs, nid_inputs, masks, lengths


def load_glove(filename, emb_dim, normalize_digits=True):
    """Loads embeddings in the glove text format in which each line is
    '<word-string> <embedding-vector>'. Dimensions of the embedding vector
    are separated with whitespace characters.

    Args:
        filename (str): Path to the embedding file.
        vocab (dict): A dictionary that maps token strings to integer index.
            Tokens not in :attr:`vocab` are not read.
        word_vecs: A 2D numpy array of shape `[vocab_size, embed_dim]`
            which is updated as reading from the file.

    Returns:
        The updated :attr:`word_vecs`.
    """
    glove_dict = dict()
    with tf.gfile.Open(filename) as fin:
        for line in fin:
            vec = line.strip().split()
            if len(vec) == 0:
                continue
            word, vec = vec[0], vec[1:]
            word = tf.compat.as_text(word)
            word = DIGIT_RE.sub("0", word) if normalize_digits else word
            glove_dict[word] = np.array([float(v) for v in vec])
            if len(vec) != emb_dim:
                raise ValueError("Inconsistent word vector sizes: %d vs %d" %
                                 (len(vec), emb_dim))
    return glove_dict


def construct_init_word_vecs(vocab, word_vecs, glove_dict):
    for word, index in vocab.items():
        if word in glove_dict:
            embedding = glove_dict[word]
        elif word.lower() in glove_dict:
            embedding = glove_dict[word.lower()]
        else: embedding = None

        if embedding is not None:
            word_vecs[index] = embedding
    return word_vecs


class CoNLLReader(object):
    def __init__(self, file_path, word_vocab, char_vocab, ner_vocab):
        self.__source_file = open(file_path, 'r', encoding='utf-8')
        self.__word_vocab = word_vocab
        self.__char_vocab = char_vocab
        self.__ner_vocab = ner_vocab

    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True):
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None

        lines = []
        while len(line.strip()) > 0:
            line = line.strip()
            lines.append(line.split(' '))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        ner_tags = []
        ner_ids = []

        for tokens in lines:
            chars = []
            char_ids = []
            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.__char_vocab[char])
            if len(chars) > MAX_CHAR_LENGTH:
                chars = chars[:MAX_CHAR_LENGTH]
                char_ids = char_ids[:MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)

            word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
            ner = tokens[4]

            words.append(word)
            word_ids.append(self.__word_vocab[word])

            ner_tags.append(ner)
            ner_ids.append(self.__ner_vocab[ner])

        return NERInstance(Sentence(words, word_ids, char_seqs, char_id_seqs), ner_tags, ner_ids)


class NERInstance(object):
    def __init__(self, sentence, ner_tags, ner_ids):
        self.sentence = sentence
        self.ner_tags = ner_tags
        self.ner_ids = ner_ids

    def length(self):
        return self.sentence.length()


class Sentence(object):
    def __init__(self, words, word_ids, char_seqs, char_id_seqs):
        self.words = words
        self.word_ids = word_ids
        self.char_seqs = char_seqs
        self.char_id_seqs = char_id_seqs

    def length(self):
        return len(self.words)
