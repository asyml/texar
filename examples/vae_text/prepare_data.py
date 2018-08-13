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
"""Utilities for downloading and preprocessing the PTB and Yohoo data.
"""
import os
import argparse

import tensorflow as tf
import texar as tx

def prepare_data(data_name):
    """Prepare datasets.
    Args:
        data_path: the path to save the data
        data_name: the name of dataset, "ptb" and "yahoo"
            are currently supported
    """
    if data_name == "ptb":
        data_path = "./simple-examples/data"
        train_path = os.path.join(data_path, "ptb.train.txt")
        if not tf.gfile.Exists(train_path):
            url = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'
            tx.data.maybe_download(url, './', extract=True)

        train_path = os.path.join(data_path, "ptb.train.txt")
        vocab_path = os.path.join(data_path, "vocab.txt")
        word_to_id = tx.data.make_vocab(
            train_path, return_type="dict")

        with open(vocab_path, 'w') as fvocab:
            for word in word_to_id:
                fvocab.write("%s\n" % word)

    elif data_name == "yahoo":
        data_path = "./data/yahoo"
        train_path = os.path.join(data_path, "yahoo.train.txt")
        if not tf.gfile.Exists(train_path):
            url = 'https://drive.google.com/file/d/'\
                  '13IsiffVjcQ-wrrbBGMwiG3sYf-DFxtXH/view?usp=sharing'
            tx.data.maybe_download(url, path='./', filenames='yahoo.zip',
                                   extract=True)
    else:
        raise ValueError('Unknown data: {}'.format(data_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare data')
    parser.add_argument('--data', type=str, help='dataset to prepare')
    args = parser.parse_args()
    prepare_data(args.data)
