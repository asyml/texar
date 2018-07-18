"""
Unit tests for utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.utils import utils


class UtilsTest(tf.test.TestCase):
    """Tests utility functions.
    """

    def test_patch_dict(self):
        """Tests :meth:`texar.core.utils.patch_dict`.
        """
        src_dict = {
            "k1": "k1",
            "k_dict_1": {
                "kd1_k1": "kd1_k1",
                "kd1_k2": "kd1_k2"
            },
            "k_dict_2": {
                "kd2_k1": "kd2_k1"
            }
        }
        tgt_dict = {
            "k1": "k1_tgt",
            "k_dict_1": {
                "kd1_k1": "kd1_k1"
            },
            "k_dict_2": "kd2_not_dict"
        }

        patched_dict = utils.patch_dict(tgt_dict, src_dict)
        self.assertEqual(patched_dict["k1"], tgt_dict["k1"])
        self.assertEqual(patched_dict["k_dict_1"], src_dict["k_dict_1"])
        self.assertEqual(patched_dict["k_dict_2"], tgt_dict["k_dict_2"])


    def test_uniquify_str(self):
        """Tests :func:`texar.core.utils.uniquify_str`.
        """
        str_set = ['str']
        unique_str = utils.uniquify_str('str', str_set)
        self.assertEqual(unique_str, 'str_1')

        str_set.append('str_1')
        str_set.append('str_2')
        unique_str = utils.uniquify_str('str', str_set)
        self.assertEqual(unique_str, 'str_3')


if __name__ == "__main__":
    tf.test.main()

