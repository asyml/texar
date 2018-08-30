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
The BLEU metric.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
from io import open # pylint: disable=redefined-builtin
import shutil
import re
import subprocess
import tempfile
import numpy as np

import tensorflow as tf

from texar.utils.dtypes import compat_as_text

# pylint: disable=too-many-locals, no-member, redefined-variable-type

__all__ = [
    "sentence_bleu_moses",
    "corpus_bleu_moses"
]

def _maybe_list_to_str(list_or_str):
    if isinstance(list_or_str, (tuple, list, np.ndarray)):
        return ' '.join(list_or_str)
    return list_or_str

def _parse_multi_bleu_ret(bleu_str, return_all=False):
    bleu_score = re.search(r"BLEU = (.+?),", bleu_str).group(1)
    bleu_score = np.float32(bleu_score)

    if return_all:
        bleus = re.search(r", (.+?)/(.+?)/(.+?)/(.+?) ", bleu_str)
        bleus = [bleus.group(group_idx) for group_idx in range(1, 5)]
        bleus = [np.float32(b) for b in bleus]
        bleu_score = [bleu_score] + bleus

    return bleu_score

def sentence_bleu_moses(references, hypothesis, lowercase=False,
                        return_all=False):
    """Calculates BLEU score of a hypothesis sentence using the
    **MOSES multi-bleu.perl** script.

    Args:
        references: A list of reference for the hypothesis.
            Each reference can be either a string, or a list of string tokens.
            List can also be numpy array.
        hypotheses: A hypothesis sentence.
            The hypothesis can be either a string, or a list of string tokens.
            List can also be numpy array.
        lowercase (bool): If `True`, pass the "-lc" flag to the multi-bleu
            script.
        return_all (bool): If `True`, returns BLEU and all n-gram precisions.

    Returns:
        If :attr:`return_all` is `False` (default), returns a float32
        BLEU score.

        If :attr:`return_all` is `True`, returns a list of 5 float32 scores:
        `[BLEU, 1-gram precision, ..., 4-gram precision]`.
    """
    return corpus_bleu_moses(
        [references], [hypothesis], lowercase=lowercase, return_all=return_all)

def corpus_bleu_moses(list_of_references, hypotheses, lowercase=False,
                      return_all=False):
    """Calculates corpus-level BLEU score using the
    **MOSES multi-bleu.perl** script.

    Args:
        list_of_references: A list of lists of references for each hypothesis.
            Each reference can be either a string, or a list of string tokens.
            List can also be numpy array.
        hypotheses: A list of hypothesis sentences.
            Each hyperthsis can be either a string, or a list of string tokens.
            List can also be numpy array.
        lowercase (bool): If `True`, pass the "-lc" flag to the multi-bleu
            script.
        return_all (bool): If `True`, returns BLEU and all n-gram precisions.

    Returns:
        If :attr:`return_all` is `False` (default), returns a float32
        BLEU score.

        If :attr:`return_all` is `True`, returns a list of 5 float32 scores:
        `[BLEU, 1-gram precision, ..., 4-gram precision]`.
    """
    list_of_references = compat_as_text(list_of_references)
    hypotheses = compat_as_text(hypotheses)

    if np.size(hypotheses) == 0:
        return np.float32(0.)   # pylint: disable=no-member

    # Get multi-bleu.perl
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    multi_bleu_path = os.path.abspath(
        os.path.join(cur_dir, "..", "..", "bin", "utils", "multi-bleu.perl"))

    # Create a temporary folder containing hyperthesis and reference files
    result_path = tempfile.mkdtemp()
    # Create hyperthesis file
    hfile_path = os.path.join(result_path, 'hyp')
    hyps = [_maybe_list_to_str(h) for h in hypotheses]
    with open(hfile_path, 'w', encoding='utf-8') as hfile:
        text = "\n".join(hyps)
        hfile.write(text)
        hfile.write("\n")
    # Create reference files
    max_nrefs = max([len(refs) for refs in list_of_references])
    rfile_path = os.path.join(result_path, 'ref')
    for rid in range(max_nrefs):
        with open(rfile_path + '%d'%rid, 'w', encoding='utf-8') as rfile:
            for refs in list_of_references:
                if rid < len(refs):
                    ref = _maybe_list_to_str(refs[rid])
                    rfile.write(ref + "\n")
                else:
                    rfile.write("\n")

    # Calculate BLEU
    multi_bleu_cmd = [multi_bleu_path]
    if lowercase:
        multi_bleu_cmd += ["-lc"]
    multi_bleu_cmd += [rfile_path]
    with open(hfile_path, "r") as hyp_input:
        try:
            multi_bleu_ret = subprocess.check_output(
                multi_bleu_cmd, stdin=hyp_input, stderr=subprocess.STDOUT)
            multi_bleu_ret = multi_bleu_ret.decode("utf-8")
            bleu_score = _parse_multi_bleu_ret(multi_bleu_ret, return_all)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                tf.logging.warning(
                    "multi-bleu.perl returned non-zero exit code")
                tf.logging.warning(error.output)
            if return_all:
                bleu_score = [np.float32(0.0)] * 5
            else:
                bleu_score = np.float32(0.0)

    shutil.rmtree(result_path)

    return np.float32(bleu_score)
