# -*- coding: utf-8 -*-
#
"""
Unit tests for utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile
import numpy as np

import tensorflow as tf

from texar.utils import utils
from texar.data.vocabulary import Vocab


class UtilsTest(tf.test.TestCase):
    """Tests utility functions.
    """

    def test_dict_patch(self):
        """Tests :meth:`texar.utils.dict_patch`.
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

        patched_dict = utils.dict_patch(tgt_dict, src_dict)
        self.assertEqual(patched_dict["k1"], tgt_dict["k1"])
        self.assertEqual(patched_dict["k_dict_1"], src_dict["k_dict_1"])
        self.assertEqual(patched_dict["k_dict_2"], tgt_dict["k_dict_2"])

    def test_strip_token(self):
        """Tests :func:`texar.utils.strip_token`
        """
        str_ = " <PAD>  <PAD>\t  i am <PAD> \t <PAD>  \t"
        self.assertEqual(utils.strip_token(str_, "<PAD>"), "i am")
        self.assertEqual(utils.strip_token(str_, ""),
                         "<PAD> <PAD> i am <PAD> <PAD>")
        self.assertEqual(utils.strip_token([str_], "<PAD>"), ["i am"])
        self.assertEqual(
            utils.strip_token(np.asarray([str_]), "<PAD>"),
            ["i am"])
        self.assertEqual(type(utils.strip_token(np.asarray([str_]), "<PAD>")),
                         np.ndarray)
        self.assertEqual(
            utils.strip_token([[[str_]], ['']], "<PAD>"),
            [[["i am"]], ['']])

        str_ = str_.split()
        self.assertEqual(utils.strip_token(str_, "<PAD>", is_token_list=True),
                         ["i", "am"])
        self.assertEqual(utils.strip_token([str_], "<PAD>", is_token_list=True),
                         [["i", "am"]])

    def test_strip_bos(self):
        """Tests :func:`texar.utils.strip_bos`
        """
        str_ = "<BOS> i am"
        self.assertEqual(utils.strip_bos(str_, "<BOS>"), "i am")
        self.assertEqual(utils.strip_bos(str_, ""), "<BOS> i am")
        self.assertEqual(utils.strip_bos([str_], "<BOS>"), ["i am"])

        str_ = str_.split()
        self.assertEqual(utils.strip_bos(str_, "<BOS>", is_token_list=True),
                         ["i", "am"])
        self.assertEqual(utils.strip_bos([str_], "<BOS>", is_token_list=True),
                         [["i", "am"]])

    def test_strip_eos(self):
        """Tests :func:`texar.utils.strip_eos`
        """
        str_ = "i am <EOS>"
        self.assertEqual(utils.strip_eos(str_, "<EOS>"), "i am")
        self.assertEqual(utils.strip_eos([str_], "<EOS>"), ["i am"])

        str_ = str_.split()
        self.assertEqual(utils.strip_eos(str_, "<EOS>", is_token_list=True),
                         ["i", "am"])
        self.assertEqual(utils.strip_eos([str_], "<EOS>", is_token_list=True),
                         [["i", "am"]])

    def test_strip_special_tokens(self):
        """Test :func:`texar.utils.strip_special_tokens`
        """
        str_ = "<BOS> i am <EOS> <PAD> <PAD>"
        self.assertEqual(utils.strip_special_tokens(str_), "i am")
        self.assertEqual(utils.strip_special_tokens([str_]), ["i am"])

        str_ = str_.split()
        self.assertEqual(utils.strip_special_tokens(str_, is_token_list=True),
                         ["i", "am"])
        self.assertEqual(utils.strip_special_tokens([str_], is_token_list=True),
                         [["i", "am"]])

    def test_str_join(self):
        """Tests :func:`texar.utils.str_join`
        """
        tokens = np.ones([2, 2, 3], dtype='str')

        str_ = utils.str_join(tokens)
        np.testing.assert_array_equal(
            str_, np.asarray([['1 1 1', '1 1 1'], ['1 1 1', '1 1 1']]))
        self.assertIsInstance(str_, np.ndarray)

        str_ = utils.str_join(tokens.tolist())
        np.testing.assert_array_equal(
            str_, [['1 1 1', '1 1 1'], ['1 1 1', '1 1 1']])
        self.assertIsInstance(str_, list)

        tokens = [[], ['1', '1']]
        str_ = utils.str_join(tokens)
        np.testing.assert_array_equal(str_, ['', '1 1'])

    def test_uniquify_str(self):
        """Tests :func:`texar.utils.uniquify_str`.
        """
        str_set = ['str']
        unique_str = utils.uniquify_str('str', str_set)
        self.assertEqual(unique_str, 'str_1')

        str_set.append('str_1')
        str_set.append('str_2')
        unique_str = utils.uniquify_str('str', str_set)
        self.assertEqual(unique_str, 'str_3')

    def test_map_ids_to_strs(self):
        """Tests :func:`texar.utils.map_ids_to_strs`.
        """
        vocab_list = ['word', '词']
        vocab_file = tempfile.NamedTemporaryFile()
        vocab_file.write('\n'.join(vocab_list).encode("utf-8"))
        vocab_file.flush()
        vocab = Vocab(vocab_file.name)

        text = [['<BOS>', 'word', '词', '<EOS>', '<PAD>'],
                ['word', '词', 'word', '词', '<PAD>']]
        text = np.asarray(text)
        ids = vocab.map_tokens_to_ids_py(text)

        ids = ids.tolist()
        text_ = utils.map_ids_to_strs(ids, vocab)

        self.assertEqual(text_[0], 'word 词')
        self.assertEqual(text_[1], 'word 词 word 词')

if __name__ == "__main__":
    tf.test.main()

