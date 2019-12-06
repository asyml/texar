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
"""Produces TFRecord files and modifies data configuration file
"""

import argparse
import logging
import os

import texar.tf as tx

from utils import data_utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task", type=str, default="MRPC",
    choices=['COLA', 'MNLI', 'MRPC', 'XNLI', 'SST'],
    help="The task to run experiment on.")
parser.add_argument(
    '--pretrained-model-name', type=str, default='bert-base-uncased',
    help="The name of a pre-trained model to load selected in the "
         "list of: `bert-base-uncased`, `bert-large-uncased`, "
         "`bert-base-cased`, `bert-large-cased`, "
         "`bert-base-multilingual-uncased`, `bert-base-multilingual-cased`, "
         "and `bert-base-chinese`.")
parser.add_argument(
    "--max-seq-length", type=int, default=128,
    help="The maxium length of sequence, longer sequence will be trimmed.")
parser.add_argument(
    "--output-dir", type=str, default=None,
    help="The output directory where the pickled files will be generated. "
         "By default it will be set to 'data/{task}'. E.g.: if "
         "task is 'MRPC', it will be set as 'data/MRPC'")
args = parser.parse_args()

logging.root.setLevel(logging.INFO)


def _modify_config_data(max_seq_length, num_train_data, num_classes):
    # Modify the data configuration file
    config_data_exists = os.path.isfile('./config_data.py')
    if config_data_exists:
        with open("./config_data.py", 'r') as file:
            filedata = file.read()
            filedata_lines = filedata.split('\n')
            idx = 0
            while True:
                if idx >= len(filedata_lines):
                    break
                line = filedata_lines[idx]
                if (line.startswith('num_classes =') or
                        line.startswith('num_train_data =') or
                        line.startswith('max_seq_length =')):
                    filedata_lines.pop(idx)
                    idx -= 1
                idx += 1

            if len(filedata_lines) > 0:
                insert_idx = 1
            else:
                insert_idx = 0
            filedata_lines.insert(
                insert_idx, '{} = {}'.format(
                    "num_train_data", num_train_data))
            filedata_lines.insert(
                insert_idx, '{} = {}'.format(
                    "num_classes", num_classes))
            filedata_lines.insert(
                insert_idx, '{} = {}'.format(
                    "max_seq_length", max_seq_length))

        with open("./config_data.py", 'w') as file:
            file.write('\n'.join(filedata_lines))
        logging.info("config_data.py has been updated")
    else:
        logging.info("config_data.py cannot be found")

    logging.info("Data preparation finished")


def main():
    """ Starts the data preparation
    """
    # Loads data
    logging.info("Loading data")

    task_datasets_rename = {
        "COLA": "CoLA",
        "SST": "SST-2",
    }

    data_dir = 'data/{}'.format(args.task)
    if args.task.upper() in task_datasets_rename:
        data_dir = 'data/{}'.format(
            task_datasets_rename[args.task])

    if args.output_dir is None:
        output_dir = data_dir
    else:
        output_dir = args.output_dir
    tx.utils.maybe_create_dir(output_dir)

    processors = {
        "COLA": data_utils.ColaProcessor,
        "MNLI": data_utils.MnliProcessor,
        "MRPC": data_utils.MrpcProcessor,
        "XNLI": data_utils.XnliProcessor,
        'SST': data_utils.SSTProcessor
    }
    processor = processors[args.task]()

    num_classes = len(processor.get_labels())
    num_train_data = len(processor.get_train_examples(data_dir))
    logging.info("num_classes: %d; num_train_data: %d",
                 num_classes, num_train_data)

    tokenizer = tx.data.BERTTokenizer(
        pretrained_model_name=args.pretrained_model_name)

    # Produces TFRecord files
    data_utils.prepare_TFRecord_data(
        processor=processor,
        tokenizer=tokenizer,
        data_dir=data_dir,
        max_seq_length=args.max_seq_length,
        output_dir=output_dir)

    _modify_config_data(args.max_seq_length, num_train_data, num_classes)


if __name__ == "__main__":
    main()
