# -*- coding: utf-8 -*-
#
"""
Unit tests for bleu.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf

from texar.evals.bleu_moses import sentence_bleu_moses, corpus_bleu_moses
from texar.evals.bleu import sentence_bleu, corpus_bleu

# pylint: disable=too-many-locals, too-many-arguments

class BLEUTest(tf.test.TestCase):
    """Tests the bleu functions.
    """

    def _test_sentence_bleu(self, references, hypothesis, lowercase,
                            true_bleu):
        bleu = sentence_bleu_moses(references=references,
                                   hypothesis=hypothesis,
                                   lowercase=lowercase)
        self.assertAlmostEqual(bleu, true_bleu, places=2)

        bleu = sentence_bleu(references=references,
                             hypothesis=hypothesis,
                             lowercase=lowercase)
        self.assertAlmostEqual(bleu, true_bleu, places=0)

    def test_sentence_strings(self):
        """Tests hypothesis as strings.
        """
        hypothesis = \
            "this is a test sentence to evaluate the good bleu score . 词"
        references = ["this is a test sentence to evaluate the bleu score ."]
        self._test_sentence_bleu(
            references, hypothesis, lowercase=False, true_bleu=67.03)

    def test_sentence_list(self):
        """Tests hypothesis as a list of tokens.
        """
        hypothesis = \
            "this is a test sentence to evaluate the good bleu score . 词"
        hypothesis = hypothesis.split()
        references = ["this is a test sentence to evaluate the bleu score ."]
        references = [references[0].split()]
        self._test_sentence_bleu(
            references, hypothesis, lowercase=False, true_bleu=67.03)

    def test_sentence_multi_references(self):
        """Tests multiple references.
        """
        hypothesis = \
            "this is a test sentence to evaluate the good bleu score . 词"
        references = ["this is a test sentence to evaluate the bleu score .",
                      "this is a test sentence to evaluate the good score ."]
        self._test_sentence_bleu(
            references, hypothesis, lowercase=False, true_bleu=76.12)

    def test_sentence_numpy(self):
        """Tests with numpy format.
        """
        hypothesis = \
            "this is a test sentence to evaluate the good bleu score . 词"
        hypothesis = np.array(hypothesis.split())
        references = ["this is a test sentence to evaluate the bleu score .",
                      "this is a test sentence to evaluate the good score ."]
        references = np.array([np.array(r.split()) for r in references])
        self._test_sentence_bleu(
            references, hypothesis, lowercase=False, true_bleu=76.12)


    def _test_corpus_bleu(self, list_of_references, hypotheses, lowercase,
                          return_all, true_bleu):
        bleu = corpus_bleu_moses(list_of_references=list_of_references,
                                 hypotheses=hypotheses,
                                 lowercase=lowercase,
                                 return_all=return_all)
        if not return_all:
            self.assertAlmostEqual(bleu, true_bleu, places=2)
        else:
            for ret, true in zip(bleu, true_bleu):
                self.assertAlmostEqual(ret, true, places=2)


        bleu = corpus_bleu(list_of_references=list_of_references,
                           hypotheses=hypotheses,
                           lowercase=lowercase,
                           return_all=return_all)
        if not return_all:
            self.assertAlmostEqual(bleu, true_bleu, places=0)
        else:
            for ret, true in zip(bleu, true_bleu):
                self.assertAlmostEqual(ret, true, places=0)


    def test_corpus_strings(self):
        """Tests corpus level BLEU.
        """
        hypotheses = [
            "this is a test sentence to evaluate the good bleu score . 词",
            "i believe that that the script is 词 perfectly correct ."
        ]
        list_of_references = [
            ["this is a test sentence to evaluate the bleu score .",
             "this is a test sentence to evaluate the good score ."],
            ["i believe that the script is perfectly correct .".split()]
        ]
        self._test_corpus_bleu(list_of_references, hypotheses,
                               False, False, 63.02)

        self._test_corpus_bleu(list_of_references, hypotheses,
                               False, True, [63.02, 87.5, 77.3, 60.0, 38.9])

if __name__ == "__main__":
    tf.test.main()
