# -*- coding: utf-8 -*-
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
"""
Utility functions related to input/output.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

# pylint: disable=invalid-name, redefined-builtin, too-many-arguments

from io import open
import os
import importlib
import yaml

import tensorflow as tf
from tensorflow import gfile

as_text = tf.compat.as_text

__all__ = [
    "load_config_single",
    "load_config",
    "write_paired_text",
    "maybe_create_dir",
    "get_files"
]

#def get_tf_logger(fname,
#                  verbosity=tf.logging.INFO,
#                  to_stdio=False,
#                  stdio_verbosity=None):
#    """Creates TF logger that allows to specify log filename and whether to
#    print to stdio at the same time.
#
#    Args:
#        fname (str): The log filename.
#        verbosity: The threshold for what messages will be logged. Default is
#            `INFO`. Other options include `DEBUG`, `ERROR`, `FATAL`, and `WARN`.
#            See :tf_main:`tf.logging <logging>`.
#        to_stdio (bool): Whether to print messages to stdio at the same time.
#        stido_verbosity (optional): The verbosity level when printing to stdio.
#            If `None` (default), the level is set to be the same as
#            :attr:`verbosity`. Ignored if :attr:`to_stdio` is False.
#
#    Returns:
#        The TF logger.
#    """

def _load_config_python(fname):
    config = {}

    config_module = importlib.import_module(fname.rstrip('.py'))
    for key in dir(config_module):
        if not (key.startswith('__') and key.endswith('__')):
            config[key] = getattr(config_module, key)

    return config

def _load_config_yaml(fname):
    with gfile.GFile(fname) as config_file:
        config = yaml.load(config_file)
    return config

def load_config_single(fname, config=None):
    """Loads config from a single file.

    The config file can be either a Python file (with suffix '.py')
    or a YAML file. If the filename is not suffixed with '.py', the file is
    parsed as YAML.

    Args:
        fname (str): The config file name.
        config (dict, optional): A config dict to which new configurations are
            added. If `None`, a new config dict is created.

    Returns:
        A `dict` of configurations.
    """
    if fname.endswith('.py'):
        new_config = _load_config_python(fname)
    else:
        new_config = _load_config_yaml(fname)

    if config is None:
        config = new_config
    else:
        for key, value in new_config.items():
            if key in config:
                if isinstance(config[key], dict):
                    config[key].update(value)
                else:
                    config[key] = value
            else:
                config[key] = value

    return config

def load_config(config_path, config=None):
    """Loads configs from (possibly multiple) file(s).

    A config file can be either a Python file (with suffix '.py')
    or a YAML file. If the filename is not suffixed with '.py', the file is
    parsed as YAML.

    Args:
        config_path: Paths to configuration files. This can be a `list` of
            config file names, or a path to a directory in which all files
            are loaded, or a string of multiple file names separated by commas.
        config (dict, optional): A config dict to which new configurations are
            added. If `None`, a new config dict is created.

    Returns:
        A `dict` of configurations.
    """
    fnames = []
    if isinstance(config_path, (list, tuple)):
        fnames = list(config_path)
    elif gfile.IsDirectory(config_path):
        for fname in gfile.ListDirectory(config_path):
            fname = os.path.join(config_path, fname)
            if not gfile.IsDirectory(fname):
                fnames.append(fname)
    else:
        for fname in config_path.split(","):
            fname = fname.strip()
            if not fname:
                continue
            fnames.append(fname)

    if config is None:
        config = {}

    for fname in fnames:
        config = load_config_single(fname, config)

    return config

# pylint: disable=too-many-locals
def write_paired_text(src, tgt, fname, append=False, mode='h', sep='\t',
                      src_fname_suffix='src', tgt_fname_suffix='tgt'):
    """Writes paired text to a file.

    Args:
        src: A list (or array) of `str` source text.
        tgt: A list (or array) of `str` target text.
        fname (str): The output filename.
        append (bool): Whether append content to the end of the file if exists.
        mode (str): The mode of writing, with the following options:

            - **'h'**: The "horizontal" mode. Each source target pair is \
                written in one line, intervened with :attr:`sep`, e.g.::

                    source_1 target_1
                    source_2 target_2

            - **'v'**: The "vertical" mode. Each source target pair is \
                written in two consecutive lines, e.g::

                    source_1
                    target_1
                    source_2
                    target_2

            - **'s'**: The "separate" mode. Each source target pair is \
                    written in corresponding lines of two files named \
                    as `"{fname}.{src_fname_suffix}"` \
                    and `"{fname}.{tgt_fname_suffix}"`, respectively.

        sep (str): The string intervening between source and target. Used
            when :attr:`mode` is set to 'h'.
        src_fname_suffix (str): Used when :attr:`mode` is 's'. The suffix to
            the source output filename. E.g., with
            `(fname='output', src_fname_suffix='src')`, the output source file
            is named as `output.src`.
        tgt_fname_suffix (str): Used when :attr:`mode` is 's'. The suffix to
            the target output filename.

    Returns:
        The fileanme(s). If `mode` == 'h' or 'v', returns
        :attr:`fname`. If `mode` == 's', returns a list of filenames
        `["{fname}.src", "{fname}.tgt"]`.
    """
    fmode = 'a' if append else 'w'
    if mode == 's':
        fn_src = '{}.{}'.format(fname, src_fname_suffix)
        fn_tgt = '{}.{}'.format(fname, tgt_fname_suffix)
        with open(fn_src, fmode, encoding='utf-8') as fs:
            fs.write(as_text('\n'.join(src)))
            fs.write('\n')
        with open(fn_tgt, fmode, encoding='utf-8') as ft:
            ft.write(as_text('\n'.join(tgt)))
            ft.write('\n')
        return fn_src, fn_tgt
    else:
        with open(fname, fmode, encoding='utf-8') as f:
            for s, t in zip(src, tgt):
                if mode == 'h':
                    text = '{}{}{}\n'.format(as_text(s), sep, as_text(t))
                    f.write(as_text(text))
                elif mode == 'v':
                    text = '{}\n{}\n'.format(as_text(s), as_text(t))
                    f.write(as_text(text))
                else:
                    raise ValueError('Unknown mode: {}'.format(mode))
        return fname

def maybe_create_dir(dirname):
    """Creates directory if doesn't exist
    """
    if not tf.gfile.IsDirectory(dirname):
        tf.gfile.MakeDirs(dirname)
        return True
    return False


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
