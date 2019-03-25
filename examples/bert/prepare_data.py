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
"""Produces TFRecords files and modifies data configuration file
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import texar as tx

# pylint: disable=no-name-in-module
from utils import data_utils, tokenization

# pylint: disable=invalid-name, too-many-locals, too-many-statements

flags = tf.flags

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "task", "mrpc",
    "The task to run experiment on. One of "
    "{'cola', 'mnli', 'mrpc', 'xnli', 'sst'}.")
flags.DEFINE_string(
    "vocab_file", 'bert_pretrained_models/uncased_L-12_H-768_A-12/vocab.txt',
    "The one-wordpiece-per-line vocabary file directory.")
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maxium length of sequence, longer sequence will be trimmed.")
flags.DEFINE_string(
    "tfrecords_output_dir", "data/tfrecords_files",
    "The output directory where the TFRecords files will be generated.")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

tf.logging.set_verbosity(tf.logging.INFO)


def prepare_data():
    """
    Builds the model and runs.
    """
    # Loads data
    tf.logging.info("Loading data")

    tx.utils.maybe_create_dir(FLAGS.tfrecords_output_dir)
    task_datasets_rename = {
        "COLA": "CoLA",
        "SST": "SST-2",
    }
    data_dir = 'data/{}'.format(FLAGS.task.upper())
    if FLAGS.task.upper() in task_datasets_rename:
        data_dir = 'data/{}'.format(
            task_datasets_rename[FLAGS.task.upper()])
    processors = {
        "cola": data_utils.ColaProcessor,
        "mnli": data_utils.MnliProcessor,
        "mrpc": data_utils.MrpcProcessor,
        "xnli": data_utils.XnliProcessor,
        'sst': data_utils.SSTProcessor
    }
    processor = processors[FLAGS.task.lower()]()

    num_classes = len(processor.get_labels())
    num_train_data = len(processor.get_train_examples(data_dir))
    tf.logging.info(
        'num_classes:%d; num_train_data:%d' % (num_classes, num_train_data))
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file,
        do_lower_case=FLAGS.do_lower_case)

    # Produces TFRecords files
    data_utils.prepare_TFRecord_data(
        processor=processor,
        tokenizer=tokenizer,
        data_dir=data_dir,
        max_seq_length=FLAGS.max_seq_length,
        output_dir=FLAGS.tfrecords_output_dir)

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
            filedata_lines.insert(
                0, '{} = {}'.format("max_seq_length", FLAGS.max_seq_length))
            filedata_lines.append(
                '{} = {}'.format("num_classes", num_classes))
            filedata_lines.append(
                '{} = {}'.format("num_train_data", num_train_data))
        with open("./config_data.py", 'w') as file:
            file.write('\n'.join(filedata_lines))
        tf.logging.info("config_data.py has been updated")
    else:
        tf.logging.info("config_data.py cannot be found")

    tf.logging.info("Data preparation finished")

def main():
    """ Starts the data preparation
    """
    prepare_data()

if __name__ == "__main__":
    main()
