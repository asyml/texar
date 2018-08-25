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
"""Preparing the SST2 dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from io import open # pylint: disable=redefined-builtin
import tensorflow as tf
import texar as tx

# pylint: disable=invalid-name, too-many-locals

flags = tf.flags

flags.DEFINE_string("data_path", "./data",
                    "Directory containing SST data. "
                    "E.g., ./data/sst2.train.sentences.txt. If not exists, "
                    "the directory will be created and SST raw data will "
                    "be downloaded.")

FLAGS = flags.FLAGS


def clean_sst_text(text):
    """Cleans tokens in the SST data, which has already been tokenized.
    """
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()

def transform_raw_sst(data_path, raw_fn, new_fn):
    """Transforms the raw data format to a new format.
    """
    fout_x_name = os.path.join(data_path, new_fn + '.sentences.txt')
    fout_x = open(fout_x_name, 'w', encoding='utf-8')
    fout_y_name = os.path.join(data_path, new_fn + '.labels.txt')
    fout_y = open(fout_y_name, 'w', encoding='utf-8')

    fin_name = os.path.join(data_path, raw_fn)
    with open(fin_name, 'r', encoding='utf-8') as fin:
        for line in fin:
            parts = line.strip().split()
            label = parts[0]
            sent = ' '.join(parts[1:])
            sent = clean_sst_text(sent)
            fout_x.write(sent + '\n')
            fout_y.write(label + '\n')

    return fout_x_name, fout_y_name

def prepare_data(data_path):
    """Preprocesses SST2 data.
    """
    train_path = os.path.join(data_path, "sst.train.sentences.txt")
    if not tf.gfile.Exists(train_path):
        url = ('https://raw.githubusercontent.com/ZhitingHu/'
               'logicnn/master/data/raw/')
        files = ['stsa.binary.phrases.train', 'stsa.binary.dev',
                 'stsa.binary.test']
        for fn in files:
            tx.data.maybe_download(url + fn, data_path, extract=True)

    fn_train, _ = transform_raw_sst(
        data_path, 'stsa.binary.phrases.train', 'sst2.train')
    transform_raw_sst(data_path, 'stsa.binary.dev', 'sst2.dev')
    transform_raw_sst(data_path, 'stsa.binary.test', 'sst2.test')

    vocab = tx.data.make_vocab(fn_train)
    fn_vocab = os.path.join(data_path, 'sst2.vocab')
    with open(fn_vocab, 'w', encoding='utf-8') as f_vocab:
        for v in vocab:
            f_vocab.write(v + '\n')

    tf.logging.info('Preprocessing done: {}'.format(data_path))

def _main(_):
    prepare_data(FLAGS.data_path)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=_main)
