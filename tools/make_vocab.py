#
"""Creates vocabulary from a set of data files.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name

import collections
import sys

import tensorflow as tf

Py3 = sys.version_info[0] == 3

flags = tf.flags

flags.DEFINE_string("files", "./",
                    "Path to the data files. Can be a pattern, e.g., "
                    "'/path/to/train*', '/path/to/train[12]'. Wrap the path "
                    "with quotation marks if a pattern is provided.")
flags.DEFINE_string("delimiter", " ", "Delimiter to split the text")
flags.DEFINE_integer("max_vocab_size", -1,
                     "Maximum size of the vocabulary. Low frequency words "
                     "that exceeding the limit will be discarded. "
                     "Set to `-1` if no truncation is wanted.")
flags.DEFINE_string("output_path", "./vocab.txt",
                    "Path of the output vocab file.")

FLAGS = flags.FLAGS

def get_files(file_paths):
    """Gets a list of file paths given possibly a pattern :attr:`file_paths`.

    Adapted from `tf.contrib.slim.data.parallel_reader.get_data_files`.

    Args:
        file_paths: A (list of) path to the files. The path can be a pattern,
            e.g., /path/to/train*, /path/to/train[12]

    Returns:
        A list of file paths.

    Raises:
        ValueError: If no files are not found
    """
    if isinstance(file_paths, (list, tuple)):
        files = []
        for f in file_paths:
            files += get_files(f)
    else:
        if '*' in file_paths or '?' in file_paths or '[' in file_paths:
            files = tf.gfile.Glob(file_paths)
        else:
            files = [file_paths]
    if not files:
        raise ValueError('No data files found in %s' % (file_paths,))
    return files

def read_words(filename, delimiter):
    """Reads word from a file.
    """
    with tf.gfile.GFile(filename, "r") as f:
        if Py3:
            return f.read().replace("\n", delimiter).split(delimiter)
        else:
            return (f.read().decode("utf-8")
                    .replace("\n", delimiter).split(delimiter))

def make_vocab(filenames, delimiter=" ", max_vocab_size=-1):
    """Builds vocab (a list of words).
    """
    words = []
    for fn in filenames:
        words += read_words(fn, delimiter)

    counter = collections.Counter(words)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    if max_vocab_size >= 0:
        words = words[:max_vocab_size]

    return words

def main(_):
    """Makes vocab.
    """
    filenames = get_files(FLAGS.files)
    vocab = make_vocab(filenames, delimiter=FLAGS.delimiter,
                       max_vocab_size=FLAGS.max_vocab_size)

    with open(FLAGS.output_path, "w") as fout:
        fout.write('\n'.join(vocab).encode("utf-8"))

if __name__ == "__main__":
    tf.app.run()
