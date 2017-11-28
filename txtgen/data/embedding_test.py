# -*- coding: utf-8 -*-
#
"""
Unit tests for embedding related operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile
import numpy as np

import tensorflow as tf

from txtgen.data import embedding


class EmbeddingTest(tf.test.TestCase):
    """Tests embedding related operations.
    """

    def test_load_glove(self):
        """Tests the load_glove function.
        """
        word_vec_lines = ["word 1.2 3.4 5.6", "词 1. 3. 5."]
        glove_file = tempfile.NamedTemporaryFile(mode="w+")
        glove_file.write('\n'.join(word_vec_lines).encode("utf-8"))
        glove_file.flush()
        vocab = {"word": 0, "词".encode('utf8'): 1}
        word_vecs = np.zeros([2, 3])

        word_vecs = embedding.load_glove(glove_file.name, vocab, word_vecs)

        self.assertEqual(word_vecs.shape[0], 2)
        self.assertEqual(word_vecs.shape[1], 3)
        np.testing.assert_array_equal(word_vecs[0], [1.2, 3.4, 5.6])
        np.testing.assert_array_equal(word_vecs[1], [1., 3., 5.])

    def test_load_word2vec(self):
        """Tests the load_word2vec function.
        """
        header = "2 3"
        words = ["word", "词"]
        vec = np.array([1.2, 3.4, 5.6], dtype='float32')
        w2v_file = tempfile.NamedTemporaryFile()
        w2v_file.write(header + "\n")
        for word in words:
            w2v_file.write((word + " ").encode("utf-8"))
            w2v_file.write(vec.tostring() + b'\n')
        w2v_file.flush()
        vocab = {"word": 0, "词".encode('utf8'): 1}
        word_vecs = np.zeros([2, 3])

        word_vecs = embedding.load_word2vec(w2v_file.name, vocab, word_vecs)

        self.assertEqual(word_vecs.shape[0], 2)
        self.assertEqual(word_vecs.shape[1], 3)
        np.testing.assert_array_equal(word_vecs[0], vec)
        np.testing.assert_array_equal(word_vecs[1], vec)


if __name__ == "__main__":
    tf.test.main()

