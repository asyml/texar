# -*- coding: utf-8 -*-
#
"""
Unit tests for vocabulary related operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile
import tensorflow as tf
import numpy as np

from txtgen.data import vocabulary


class VocabularyTest(tf.test.TestCase):
    """Tests vocabulary related operations.
    """

    def test_make_defaultdict(self):
        """Tests the _make_defaultdict function.
        """
        keys = ['word', '词']
        values = [0, 1]
        default_value = -1

        dict_ = vocabulary._make_defaultdict(keys, values, default_value)

        self.assertEqual(len(dict_), 2)
        self.assertEqual(dict_['word'], 0)
        self.assertEqual(dict_['词'], 1)
        self.assertEqual(dict_['sth_else'], -1)


if __name__ == "__main__":
    tf.test.main()

