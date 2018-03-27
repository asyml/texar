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
from six.moves import urllib

import tensorflow as tf

__all__ = [
]

def create_dir_if_needed(dirname):
    """Creates directory if doesn't exist
    """
    if not tf.gfile.IsDirectory(dirname):
        tf.gfile.MakeDirs(dirname)

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
