# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications copyright (C) 2018 Texar
# ==============================================================================
"""
Python implementation of BLEU and smoothed BLEU adapted from:
    `https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py`

This module provides a Python implementation of BLEU and smoothed BLEU.
Smooth BLEU is computed following the method outlined in the paper:

    (Lin et al. 2004) ORANGE: a method for evaluating automatic evaluation
    metrics for maching translation.
    Chin-Yew Lin, Franz Josef Och. COLING 2004.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import collections
import math

from texar.utils.dtypes import compat_as_text, is_str

# pylint: disable=invalid-name, too-many-branches, too-many-locals
# pylint: disable=too-many-arguments

__all__ = [
    "sentence_bleu",
    "corpus_bleu"
]

def _get_ngrams(segment, max_order):
    """Extracts all n-grams up to a given maximum order from an input segment.

    Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this
            methods.

    Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i+order])
            ngram_counts[ngram] += 1
    return ngram_counts

def _maybe_str_to_list(list_or_str):
    if is_str(list_or_str):
        return list_or_str.split()
    return list_or_str

def _lowercase(str_list):
    return [str_.lower() for str_ in str_list]

def sentence_bleu(references, hypothesis, max_order=4, lowercase=False,
                  smooth=False, return_all=False):
    """Calculates BLEU score of a hypothesis sentence.

    Args:
        references: A list of reference for the hypothesis.
            Each reference can be either a list of string tokens, or a string
            containing tokenized tokens separated with whitespaces.
            List can also be numpy array.
        hypotheses: A hypothesis sentence.
            Each hypothesis can be either a list of string tokens, or a
            string containing tokenized tokens separated with whitespaces.
            List can also be numpy array.
        lowercase (bool): If `True`, lowercase reference and hypothesis tokens.
        max_order (int): Maximum n-gram order to use when computing BLEU score.
        smooth (bool): Whether or not to apply (Lin et al. 2004) smoothing.
        return_all (bool): If `True`, returns BLEU and all n-gram precisions.

    Returns:
        If :attr:`return_all` is `False` (default), returns a float32
        BLEU score.

        If :attr:`return_all` is `True`, returns a list of float32 scores:
        `[BLEU] + n-gram precisions`, which is of length :attr:`max_order`+1.
    """
    return corpus_bleu(
        [references], [hypothesis], max_order=max_order, lowercase=lowercase,
        smooth=smooth, return_all=return_all)

def corpus_bleu(list_of_references, hypotheses, max_order=4, lowercase=False,
                smooth=False, return_all=True):
    """Computes corpus-level BLEU score.

    Args:
        list_of_references: A list of lists of references for each hypothesis.
            Each reference can be either a list of string tokens, or a string
            containing tokenized tokens separated with whitespaces.
            List can also be numpy array.
        hypotheses: A list of hypothesis sentences.
            Each hypothesis can be either a list of string tokens, or a
            string containing tokenized tokens separated with whitespaces.
            List can also be numpy array.
        lowercase (bool): If `True`, lowercase reference and hypothesis tokens.
        max_order (int): Maximum n-gram order to use when computing BLEU score.
        smooth (bool): Whether or not to apply (Lin et al. 2004) smoothing.
        return_all (bool): If `True`, returns BLEU and all n-gram precisions.

    Returns:
        If :attr:`return_all` is `False` (default), returns a float32
        BLEU score.

        If :attr:`return_all` is `True`, returns a list of float32 scores:
        `[BLEU] + n-gram precisions`, which is of length :attr:`max_order`+1.
    """
    list_of_references = compat_as_text(list_of_references)
    hypotheses = compat_as_text(hypotheses)

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    hyperthsis_length = 0
    for (references, hyperthsis) in zip(list_of_references, hypotheses):
        reference_length += min(len(r) for r in references)
        hyperthsis_length += len(hyperthsis)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            reference = _maybe_str_to_list(reference)
            if lowercase:
                reference = _lowercase(reference)
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)

        hyperthsis = _maybe_str_to_list(hyperthsis)
        if lowercase:
            hyperthsis = _lowercase(hyperthsis)
        hyperthsis_ngram_counts = _get_ngrams(hyperthsis, max_order)

        overlap = hyperthsis_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]
        for order in range(1, max_order+1):
            possible_matches = len(hyperthsis) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(hyperthsis_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        try:
            bp = math.exp(1 - 1. / ratio)
        except ZeroDivisionError:
            bp = math.exp(1 - 1. / (ratio + 1e-8))

    bleu = geo_mean * bp

    if return_all:
        return [bleu * 100] + [p * 100 for p in precisions]
    else:
        return bleu * 100
