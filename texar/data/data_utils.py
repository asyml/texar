#
"""
Various utilities specific to data processing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import tarfile
import zipfile
import logging
import collections
from six.moves import urllib

import tensorflow as tf

# pylint: disable=invalid-name

__all__ = [
    "create_dir_if_needed",
    "maybe_download"
]

Py3 = sys.version_info[0] == 3

def create_dir_if_needed(dirname):
    """Creates directory if doesn't exist
    """
    if not tf.gfile.IsDirectory(dirname):
        tf.gfile.MakeDirs(dirname)
        return True
    return False

def maybe_download(urls, path, extract=False):
    """Downloads a set of files.

    Args:
        urls: A (list of) urls to download files.
        path (str): The destination path to save the files.
        extract (bool): Whether to extract compressed files.

    Returns:
        A list of paths to the downloaded files.
    """
    create_dir_if_needed(path)

    if not isinstance(urls, (list, tuple)):
        urls = [urls]
    result = []
    for url in urls:
        filename = url.split('/')[-1]
        # If downloading from GitHub, remove suffix ?raw=True
        # from local filename
        if filename.endswith("?raw=true"):
            filename = filename[:-9]

        filepath = os.path.join(path, filename)
        result.append(filepath)

        if not tf.gfile.Exists(filepath):
            def _progress(count, block_size, total_size):
                percent = float(count * block_size) / float(total_size) * 100.
                # pylint: disable=cell-var-from-loop
                sys.stdout.write('\r>> Downloading %s %.1f%%' %
                                 (filename, percent))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded {} {} bytes.'.format(
                filename, statinfo.st_size))

            if extract:
                logging.info('Extract %s', filepath)
                if tarfile.is_tarfile(filepath):
                    tarfile.open(filepath, 'r').extractall(path)
                elif zipfile.is_zipfile(filepath):
                    with zipfile.ZipFile(filepath) as zfile:
                        zfile.extractall(path)
                else:
                    logging.info("Unknown compression type. Only .tar.gz, "
                                 ".tar.bz2, .tar, and .zip are supported")

    return result

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

def read_words(filename, delimiter, newline_token="\n"):
    """Reads word from a file.

    Args:
        filename (str): Path to the file.
        delimiter (str): Delimiter to split the string into tokens.
        newline_token (str): The token to replace the original newline
            token `\n`. For example, `newline_token=tx.data.SpecialTokens.EOS`.
    """
    with tf.gfile.GFile(filename, "r") as f:
        if Py3:
            return f.read().replace("\n", newline_token).split(delimiter)
        else:
            return (f.read().decode("utf-8")
                    .replace("\n", newline_token).split(delimiter))

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
