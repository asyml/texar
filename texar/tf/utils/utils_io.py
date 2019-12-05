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
Utility functions related to input/output.
"""

import os

__all__ = [
    "write_paired_text",
    "maybe_create_dir",
]


def write_paired_text(src, tgt, fname, append=False, mode='h', sep='\t',
                      src_fname_suffix='src', tgt_fname_suffix='tgt'):
    r"""Writes paired text to a file.

    Args:
        src: A list (or array) of ``str`` source text.
        tgt: A list (or array) of ``str`` target text.
        fname (str): The output filename.
        append (bool): Whether append content to the end of the file if exists.
        mode (str): The mode of writing, with the following options:

            - **'h'**: The "horizontal" mode. Each source target pair is
              written in one line, intervened with :attr:`sep`, e.g.::

                  source_1 target_1
                  source_2 target_2

            - **'v'**: The ``"vertical"`` mode. Each source target pair is
              written in two consecutive lines, e.g::

                  source_1
                  target_1
                  source_2
                  target_2

            - **'s'**: The "separate" mode. Each source target pair is
              written in corresponding lines of two files named
              as ``"{fname}.{src_fname_suffix}"``
              and ``"{fname}.{tgt_fname_suffix}"``, respectively.

        sep (str): The string intervening between source and target. Used
            when :attr:`mode` is set to ``"h"``.
        src_fname_suffix (str): Used when :attr:`mode` is ``"s"``. The suffix
            to the source output filename. For example, with
            ``(fname='output', src_fname_suffix='src')``, the output source
            file is named as ``output.src``.
        tgt_fname_suffix (str): Used when :attr:`mode` is ``"s"``. The suffix
            to the target output filename.

    Returns:
        The filename(s). If ``mode`` == ``"h"`` or ``"v"``, returns
        :attr:`fname`. If ``mode`` == ``"s"``, returns a list of filenames
        ``["{fname}.src", "{fname}.tgt"]``.
    """
    fmode = 'a' if append else 'w'
    if mode == 's':
        fn_src = '{}.{}'.format(fname, src_fname_suffix)
        fn_tgt = '{}.{}'.format(fname, tgt_fname_suffix)
        with open(fn_src, fmode, encoding='utf-8') as fs:
            fs.write('\n'.join(src))
            fs.write('\n')
        with open(fn_tgt, fmode, encoding='utf-8') as ft:
            ft.write('\n'.join(tgt))
            ft.write('\n')
        return fn_src, fn_tgt
    else:
        with open(fname, fmode, encoding='utf-8') as f:
            for s, t in zip(src, tgt):
                if mode == 'h':
                    text = '{}{}{}\n'.format(s, sep, t)
                    f.write(text)
                elif mode == 'v':
                    text = '{}\n{}\n'.format(s, t)
                    f.write(text)
                else:
                    raise ValueError('Unknown mode: {}'.format(mode))
        return fname


def maybe_create_dir(dirname: str) -> bool:
    r"""Creates directory if it does not exist.

    Args:
        dirname (str): Path to the directory.

    Returns:
        bool: Whether a new directory is created.
    """
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
        return True
    return False
