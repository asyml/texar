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
"""
Various utilities specific to data processing.
"""

import logging
import os
import sys
import tarfile
import urllib.request
import zipfile

import numpy as np

from texar.tf.utils import utils_io
from texar.tf.utils.types import MaybeList


__all__ = [
    "maybe_download",
    "count_file_lines",
]


def maybe_download(urls, path, filenames=None, extract=False):
    r"""Downloads a set of files.

    Args:
        urls: A (list of) URLs to download files.
        path (str): The destination path to save the files.
        filenames: A (list of) strings of the file names. If given,
            must have the same length with :attr:`urls`. If `None`,
            filenames are extracted from :attr:`urls`.
        extract (bool): Whether to extract compressed files.

    Returns:
        A list of paths to the downloaded files.
    """
    utils_io.maybe_create_dir(path)

    if not isinstance(urls, (list, tuple)):
        is_list = False
        urls = [urls]
    else:
        is_list = True
    if filenames is not None:
        if not isinstance(filenames, (list, tuple)):
            filenames = [filenames]
        if len(urls) != len(filenames):
            raise ValueError(
                '`filenames` must have the same number of elements as `urls`.')

    result = []
    for i, url in enumerate(urls):
        if filenames is not None:
            filename = filenames[i]
        elif 'drive.google.com' in url:
            filename = _extract_google_drive_file_id(url)
        else:
            filename = url.split('/')[-1]
            # If downloading from GitHub, remove suffix ?raw=True
            # from local filename
            if filename.endswith("?raw=true"):
                filename = filename[:-9]

        filepath = os.path.join(path, filename)
        result.append(filepath)

        # if not tf.gfile.Exists(filepath):
        if not os.path.exists(filepath):
            if 'drive.google.com' in url:
                filepath = _download_from_google_drive(url, filename, path)
            else:
                filepath = _download(url, filename, path)

            if extract:
                logging.info('Extract %s', filepath)
                if tarfile.is_tarfile(filepath):
                    tarfile.open(filepath, 'r').extractall(path)
                elif zipfile.is_zipfile(filepath):
                    with zipfile.ZipFile(filepath) as zfile:
                        zfile.extractall(path)
                else:
                    logging.info("Unknown compression type. Only .tar.gz"
                                 ".tar.bz2, .tar, and .zip are supported")
    if not is_list:
        return result[0]
    return result


def _download(url: str, filename: str, path: str) -> str:
    def _progress_hook(count, block_size, total_size):
        percent = float(count * block_size) / float(total_size) * 100.
        sys.stdout.write(f'\r>> Downloading {filename} {percent:.1f}%')
        sys.stdout.flush()

    filepath = os.path.join(path, filename)
    filepath, _ = urllib.request.urlretrieve(url, filepath, _progress_hook)
    print()
    statinfo = os.stat(filepath)
    print(f'Successfully downloaded {filename} {statinfo.st_size} bytes')

    return filepath


def _extract_google_drive_file_id(url: str) -> str:
    # id is between `/d/` and '/'
    url_suffix = url[url.find('/d/') + 3:]
    file_id = url_suffix[:url_suffix.find('/')]
    return file_id


def _download_from_google_drive(url: str, filename: str, path: str) -> str:
    r"""Adapted from `https://github.com/saurabhshri/gdrive-downloader`
    """

    try:
        import requests
    except ImportError:
        print("The requests library must be installed to download files from "
              "Google drive. Please see: https://github.com/psf/requests")
        raise

    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    file_id = _extract_google_drive_file_id(url)

    gurl = "https://docs.google.com/uc?export=download"
    sess = requests.Session()
    response = sess.get(gurl, params={'id': file_id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = sess.get(gurl, params=params, stream=True)

    filepath = os.path.join(path, filename)
    CHUNK_SIZE = 32768
    with open(filepath, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

    print(f'Successfully downloaded {filename}')

    return filepath


def count_file_lines(filenames: MaybeList[str]) -> int:
    r"""Counts the number of lines in the file(s).
    """

    def _count_lines(fn):
        with open(fn, "rb") as f:
            i = -1
            for i, _ in enumerate(f):
                pass
            return i + 1

    if not isinstance(filenames, (list, tuple)):
        filenames = [filenames]
    num_lines = np.sum([_count_lines(fn) for fn in filenames]).item()
    return num_lines
