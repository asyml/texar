# -*- coding: utf-8 -*-
#
"""
Unit tests for data utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile

import tensorflow as tf

from texar.data import data_utils


class CountFileLinesTest(tf.test.TestCase):
    """Tests :func:`texar.data.data_utils.count_file_lines`.
    """

    def test_load_glove(self):
        """Tests the load_glove function.
        """
        file_1 = tempfile.NamedTemporaryFile(mode="w+")
        num_lines = data_utils.count_file_lines(file_1.name)
        self.assertEqual(num_lines, 0)

        file_2 = tempfile.NamedTemporaryFile(mode="w+")
        file_2.write('\n'.join(['x']*5))
        file_2.flush()
        num_lines = data_utils.count_file_lines(
            [file_1.name, file_2.name, file_2.name])
        self.assertEqual(num_lines, 0+5+5)


if __name__ == "__main__":
    tf.test.main()

