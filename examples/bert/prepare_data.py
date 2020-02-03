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

import os
import tensorflow as tf
import texar.tf as tx

# pylint: disable=no-name-in-module
from utils import data_utils

# pylint: disable=invalid-name, too-many-locals, too-many-statements

flags = tf.flags

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "task", "MRPC",
    "The task to run experiment on. One of "
    "{'COLA', 'MNLI', 'MRPC', 'XNLI', 'SST'}.")
flags.DEFINE_string(
    "pretrained_model_name", 'bert-base-uncased',
    "The name of pre-trained BERT model. See the doc of "
    "`texar.tf.modules.PretrainedBERTMixin for all supported models.`")
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum length of sequence, longer sequence will be trimmed.")
flags.DEFINE_string(
    "tfrecord_output_dir", None,
    "The output directory where the TFRecord files will be generated. "
    "By default it will be set to 'data/{task}'. E.g.: if "
    "task is 'MRPC', it will be set as 'data/MRPC'")

tf.logging.set_verbosity(tf.logging.INFO)


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
        tf.logging.info("config_data.py has been updated")
    else:
        tf.logging.info("config_data.py cannot be found")

    tf.logging.info("Data preparation finished")


def main():
    """Prepares data.
    """
    # Loads data
    tf.logging.info("Loading data")

    task_datasets_rename = {
        "COLA": "CoLA",
        "SST": "SST-2",
    }

    data_dir = 'data/{}'.format(FLAGS.task)
    if FLAGS.task.upper() in task_datasets_rename:
        data_dir = 'data/{}'.format(
            task_datasets_rename[FLAGS.task])

    if FLAGS.tfrecord_output_dir is None:
        tfrecord_output_dir = data_dir
    else:
        tfrecord_output_dir = FLAGS.tfrecord_output_dir
    tx.utils.maybe_create_dir(tfrecord_output_dir)

    processors = {
        "COLA": data_utils.ColaProcessor,
        "MNLI": data_utils.MnliProcessor,
        "MRPC": data_utils.MrpcProcessor,
        "XNLI": data_utils.XnliProcessor,
        'SST': data_utils.SSTProcessor
    }
    processor = processors[FLAGS.task]()

    num_classes = len(processor.get_labels())
    num_train_data = len(processor.get_train_examples(data_dir))
    tf.logging.info(
        'num_classes:%d; num_train_data:%d' % (num_classes, num_train_data))

    tokenizer = tx.data.BERTTokenizer(
        pretrained_model_name=FLAGS.pretrained_model_name)

    # Produces TFRecord files
    data_utils.prepare_TFRecord_data(
        processor=processor,
        tokenizer=tokenizer,
        data_dir=data_dir,
        max_seq_length=FLAGS.max_seq_length,
        output_dir=tfrecord_output_dir)

    _modify_config_data(FLAGS.max_seq_length, num_train_data, num_classes)


if __name__ == "__main__":
    main()
