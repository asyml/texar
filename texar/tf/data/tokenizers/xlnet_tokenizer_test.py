"""
Unit tests for pre-trained XLNet tokenizer.
"""

import os
import pickle
import tempfile

import tensorflow as tf

from texar.tf.data.data_utils import maybe_download
from texar.tf.data.tokenizers.xlnet_tokenizer import \
    XLNetTokenizer, SPIECE_UNDERLINE
from texar.tf.utils.test import pretrained_test


class XLNetTokenizerTest(tf.test.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        # Use the test sentencepiece model downloaded from huggingface
        # transformers
        self.SAMPLE_VOCAB = maybe_download(
            'https://github.com/huggingface/transformers/blob/master/'
            'transformers/tests/fixtures/test_sentencepiece.model?raw=true',
            self.tmp_dir.name)

        self.tokenizer = XLNetTokenizer.load(
            self.SAMPLE_VOCAB[0], configs={'keep_accents': True})
        self.tokenizer.save(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    @pretrained_test
    def test_model_loading(self):
        for pretrained_model_name in \
                XLNetTokenizer.available_checkpoints():
            tokenizer = XLNetTokenizer(
                pretrained_model_name=pretrained_model_name)
            _ = tokenizer.map_text_to_token(u"This is a test")

    def test_tokenize(self):
        tokens = self.tokenizer.map_text_to_token(u'This is a test')
        self.assertListEqual(tokens, [u'▁This', u'▁is', u'▁a', u'▁t', u'est'])

        self.assertListEqual(
            self.tokenizer.map_token_to_id(tokens),
            [285, 46, 10, 170, 382])

        tokens = self.tokenizer.map_text_to_token(
            u"I was born in 92000, and this is falsé.")
        self.assertListEqual(tokens, [SPIECE_UNDERLINE + u'I',
                                      SPIECE_UNDERLINE + u'was',
                                      SPIECE_UNDERLINE + u'b',
                                      u'or', u'n', SPIECE_UNDERLINE + u'in',
                                      SPIECE_UNDERLINE + u'',
                                      u'9', u'2', u'0', u'0', u'0', u',',
                                      SPIECE_UNDERLINE + u'and',
                                      SPIECE_UNDERLINE + u'this',
                                      SPIECE_UNDERLINE + u'is',
                                      SPIECE_UNDERLINE + u'f', u'al', u's',
                                      u'é', u'.'])
        ids = self.tokenizer.map_token_to_id(tokens)
        self.assertListEqual(
            ids, [8, 21, 84, 55, 24, 19, 7, 0,
                  602, 347, 347, 347, 3, 12, 66,
                  46, 72, 80, 6, 0, 4])

        back_tokens = self.tokenizer.map_id_to_token(ids)
        self.assertListEqual(back_tokens, [SPIECE_UNDERLINE + u'I',
                                           SPIECE_UNDERLINE + u'was',
                                           SPIECE_UNDERLINE + u'b',
                                           u'or', u'n',
                                           SPIECE_UNDERLINE + u'in',
                                           SPIECE_UNDERLINE + u'', u'<unk>',
                                           u'2', u'0', u'0', u'0', u',',
                                           SPIECE_UNDERLINE + u'and',
                                           SPIECE_UNDERLINE + u'this',
                                           SPIECE_UNDERLINE + u'is',
                                           SPIECE_UNDERLINE + u'f', u'al', u's',
                                           u'<unk>', u'.'])

    def test_pickle(self):
        tokenizer = XLNetTokenizer.load(self.tmp_dir.name)
        self.assertIsNotNone(tokenizer)

        text = u"Munich and Berlin are nice cities"
        subwords = tokenizer.map_text_to_token(text)

        with tempfile.TemporaryDirectory() as tmpdirname:
            filename = os.path.join(tmpdirname, u"tokenizer.bin")
            with open(filename, "wb") as f:
                pickle.dump(tokenizer, f)
            with open(filename, "rb") as f:
                tokenizer_new = pickle.load(f)

        subwords_loaded = tokenizer_new.map_text_to_token(text)

        self.assertListEqual(subwords, subwords_loaded)

    def test_save_load(self):
        tokenizer = XLNetTokenizer.load(self.tmp_dir.name)

        before_tokens = tokenizer.map_text_to_id(
            u"He is very happy, UNwant\u00E9d,running")

        with tempfile.TemporaryDirectory() as tmpdirname:
            tokenizer.save(tmpdirname)
            tokenizer = tokenizer.load(tmpdirname)

        after_tokens = tokenizer.map_text_to_id(
            u"He is very happy, UNwant\u00E9d,running")
        self.assertListEqual(before_tokens, after_tokens)

    def test_pretrained_model_list(self):
        model_list_1 = list(XLNetTokenizer._MODEL2URL.keys())
        model_list_2 = list(XLNetTokenizer._MAX_INPUT_SIZE.keys())

        self.assertListEqual(model_list_1, model_list_2)

    def test_encode_decode(self):
        tokenizer = XLNetTokenizer.load(self.tmp_dir.name)

        input_text = u"This is a test"
        output_text = u"This is a test"

        tokens = tokenizer.map_text_to_token(input_text)
        ids = tokenizer.map_token_to_id(tokens)
        ids_2 = tokenizer.map_text_to_id(input_text)
        self.assertListEqual(ids, ids_2)

        tokens_2 = tokenizer.map_id_to_token(ids)
        text_2 = tokenizer.map_id_to_text(ids)

        self.assertEqual(text_2, output_text)

        self.assertNotEqual(len(tokens_2), 0)
        self.assertIsInstance(text_2, str)

    def test_add_tokens(self):
        tokenizer = XLNetTokenizer.load(self.tmp_dir.name)

        vocab_size = tokenizer.vocab_size
        all_size = len(tokenizer)

        self.assertNotEqual(vocab_size, 0)
        self.assertEqual(vocab_size, all_size)

        new_toks = ["aaaaabbbbbb", "cccccccccdddddddd"]
        added_toks = tokenizer.add_tokens(new_toks)
        vocab_size_2 = tokenizer.vocab_size
        all_size_2 = len(tokenizer)

        self.assertNotEqual(vocab_size_2, 0)
        self.assertEqual(vocab_size, vocab_size_2)
        self.assertEqual(added_toks, len(new_toks))
        self.assertEqual(all_size_2, all_size + len(new_toks))

        tokens = tokenizer.map_text_to_id("aaaaabbbbbb low cccccccccdddddddd l")
        self.assertGreaterEqual(len(tokens), 4)
        self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
        self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)

        new_toks_2 = {'eos_token': ">>>>|||<||<<|<<",
                      'pad_token': "<<<<<|||>|>>>>|>"}
        added_toks_2 = tokenizer.add_special_tokens(new_toks_2)
        vocab_size_3 = tokenizer.vocab_size
        all_size_3 = len(tokenizer)

        self.assertNotEqual(vocab_size_3, 0)
        self.assertEqual(vocab_size, vocab_size_3)
        self.assertEqual(added_toks_2, len(new_toks_2))
        self.assertEqual(all_size_3, all_size_2 + len(new_toks_2))

        tokens = tokenizer.map_text_to_id(
            ">>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd "
            "<<<<<|||>|>>>>|> l")

        self.assertGreaterEqual(len(tokens), 6)
        self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
        self.assertGreater(tokens[0], tokens[1])
        self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)
        self.assertGreater(tokens[-2], tokens[-3])
        self.assertEqual(tokens[0],
                         tokenizer.map_token_to_id(tokenizer.eos_token))
        self.assertEqual(tokens[-2],
                         tokenizer.map_token_to_id(tokenizer.pad_token))

    def test_tokenizer_lower(self):
        tokenizer = XLNetTokenizer.load(
            self.SAMPLE_VOCAB[0], configs={'do_lower_case': True,
                                           'keep_accents': False})
        tokens = tokenizer.map_text_to_token(
            u"I was born in 92000, and this is falsé.")
        self.assertListEqual(tokens, [SPIECE_UNDERLINE + u'', u'i',
                                      SPIECE_UNDERLINE + u'was',
                                      SPIECE_UNDERLINE + u'b',
                                      u'or', u'n', SPIECE_UNDERLINE + u'in',
                                      SPIECE_UNDERLINE + u'',
                                      u'9', u'2', u'0', u'0', u'0', u',',
                                      SPIECE_UNDERLINE + u'and',
                                      SPIECE_UNDERLINE + u'this',
                                      SPIECE_UNDERLINE + u'is',
                                      SPIECE_UNDERLINE + u'f', u'al', u'se',
                                      u'.'])
        self.assertListEqual(tokenizer.map_text_to_token(u"H\u00E9llo"),
                             [u"▁he", u"ll", u"o"])

    def test_tokenizer_no_lower(self):
        tokenizer = XLNetTokenizer.load(
            self.SAMPLE_VOCAB[0], configs={'do_lower_case': False,
                                           'keep_accents': False})
        tokens = tokenizer.map_text_to_token(
            u"I was born in 92000, and this is falsé.")
        self.assertListEqual(tokens, [SPIECE_UNDERLINE + u'I',
                                      SPIECE_UNDERLINE + u'was',
                                      SPIECE_UNDERLINE + u'b', u'or',
                                      u'n', SPIECE_UNDERLINE + u'in',
                                      SPIECE_UNDERLINE + u'',
                                      u'9', u'2', u'0', u'0', u'0', u',',
                                      SPIECE_UNDERLINE + u'and',
                                      SPIECE_UNDERLINE + u'this',
                                      SPIECE_UNDERLINE + u'is',
                                      SPIECE_UNDERLINE + u'f', u'al', u'se',
                                      u'.'])

    def test_encode_text(self):
        text_1 = u"He is very happy"
        text_2 = u"unwanted, running"

        text_1_ids = self.tokenizer.map_text_to_id(text_1)
        text_2_ids = self.tokenizer.map_text_to_id(text_2)

        cls_token_id = self.tokenizer.map_token_to_id(self.tokenizer.cls_token)
        sep_token_id = self.tokenizer.map_token_to_id(self.tokenizer.sep_token)

        input_ids, segment_ids, input_mask = \
            self.tokenizer.encode_text(text_1, None, 4)

        self.assertListEqual(input_ids,
                             text_1_ids[:2] + [sep_token_id] + [cls_token_id])
        self.assertListEqual(segment_ids, [0, 0, 0, 2])
        self.assertListEqual(input_mask, [0, 0, 0, 0])

        input_ids, segment_ids, input_mask = \
            self.tokenizer.encode_text(text_1, text_2, 7)

        self.assertListEqual(input_ids, text_1_ids[:2] +
                             [sep_token_id] + text_2_ids[:2] + [sep_token_id] +
                             [cls_token_id])
        self.assertListEqual(segment_ids, [0, 0, 0, 1, 1, 1, 2])
        self.assertListEqual(input_mask, [0, 0, 0, 0, 0, 0, 0])


if __name__ == "__main__":
    tf.test.main()
