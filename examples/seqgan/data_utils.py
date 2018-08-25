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
"""SeqGAN for language modeling
"""
import os
import argparse
import importlib
import tensorflow as tf
import texar as tx

parser = argparse.ArgumentParser(description='prepare data')
parser.add_argument('--dataset', type=str, default='ptb',
                    help='dataset to prepare')
parser.add_argument('--data_path', type=str, default='./',
                    help="Directory containing coco. If not exists, "
                    "the directory will be created, and the data "
                    "will be downloaded.")
parser.add_argument('--config', type=str, default='config_ptb_small',
                    help='The config to use.')
args = parser.parse_args()

config = importlib.import_module(args.config)


def prepare_data(args, config, train_path):
    """Downloads the PTB or COCO dataset
    """
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)

    ptb_url = 'https://jxhe.github.io/download/ptb_data.tgz'
    coco_url = 'https://VegB.github.io/downloads/coco_data.tgz'

    data_path = args.data_path

    if not tf.gfile.Exists(train_path):
        url = ptb_url if args.dataset == 'ptb' else coco_url
        tx.data.maybe_download(url, data_path, extract=True)
        os.remove('%s_data.tgz' % args.dataset)


if __name__ == '__main__':
    prepare_data(args, config, config.train_data_hparams['dataset']['files'])
