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
"""Creates vocabulary from a set of data files.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name

import sys

import tensorflow as tf

import texar as tx

Py3 = sys.version_info[0] == 3

flags = tf.flags

flags.DEFINE_string("files", "./train.txt",
                    "Path to the data files. Can be a pattern, e.g., "
                    "'/path/to/train*', '/path/to/train[12]'. Wrap the path "
                    "with quotation marks if a pattern is provided.")
flags.DEFINE_integer("max_vocab_size", -1,
                     "Maximum size of the vocabulary. Low frequency words "
                     "that exceeding the limit will be discarded. "
                     "Set to `-1` if no truncation is wanted.")
flags.DEFINE_string("output_path", "./vocab.txt",
                    "Path of the output vocab file.")
flags.DEFINE_string("newline_token", None,
                    "The token to replace the original newline token '\n'. "
                    "For example, `--newline_token '<EOS>'`. If not "
                    "specified, no replacement is performed.")

FLAGS = flags.FLAGS


def main(_):
    """Makes vocab.
    """
    filenames = tx.utils.get_files(FLAGS.files)
    vocab = tx.data.make_vocab(filenames,
                               max_vocab_size=FLAGS.max_vocab_size,
                               newline_token=FLAGS.newline_token)

    with open(FLAGS.output_path, "w") as fout:
        fout.write('\n'.join(vocab).encode("utf-8"))

if __name__ == "__main__":
    tf.app.run()
