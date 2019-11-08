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
Pre-trained GPT-2 tokenizer.

Code structure adapted from:
    `https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/tokenization_gpt2.py`
"""

from typing import Any, Dict, List, Optional, Tuple

import os
import json
import regex as re

from texar.tf.modules.pretrained.gpt2 import PretrainedGPT2Mixin
from texar.tf.data.tokenizers.tokenizer_base import TokenizerBase
from texar.tf.data.tokenizers.gpt2_tokenizer_utils import \
    bytes_to_unicode, get_pairs

__all__ = [
    'GPT2Tokenizer',
]


class GPT2Tokenizer(TokenizerBase, PretrainedGPT2Mixin):
    r"""Pre-trained GPT2 Tokenizer.

    Args:
        pretrained_model_name (optional): a `str`, the name of
            pre-trained model (e.g., `117M`). Please refer to
            :class:`~texar.torch.modules.PretrainedGPT2Mixin` for
            all supported models.
            If None, the model name in :attr:`hparams` is used.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory (``texar_data`` folder under user's home
            directory) will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.
    """

    _IS_PRETRAINED = True
    _MAX_INPUT_SIZE = {
        'gpt2-small': 1024,
        'gpt2-medium': 1024,
        'gpt2-large': 1024,
        'gpt2-xl': 1024,
    }
    _DEPRECATED_MAX_INPUT_SIZE = {
        '117M': 1024,
        '345M': 1024,
    }
    _MAX_INPUT_SIZE.update(_DEPRECATED_MAX_INPUT_SIZE)

    _VOCAB_FILE_NAMES = {
        'vocab_file': 'encoder.json',
        'merges_file': 'vocab.bpe',
    }

    def __init__(self,
                 pretrained_model_name: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 hparams=None):
        self.load_pretrained_config(pretrained_model_name, cache_dir, hparams)

        super().__init__(hparams=None)

        self.config = {
            'errors': self.hparams['errors']
        }

        if self.pretrained_model_dir is not None:
            vocab_file = os.path.join(self.pretrained_model_dir,
                                      self._VOCAB_FILE_NAMES['vocab_file'])
            merges_file = os.path.join(self.pretrained_model_dir,
                                       self._VOCAB_FILE_NAMES['merges_file'])
            assert pretrained_model_name is not None
            if self._MAX_INPUT_SIZE.get(pretrained_model_name):
                self.max_len = self._MAX_INPUT_SIZE[pretrained_model_name]
        else:
            vocab_file = self.hparams['vocab_file']
            merges_file = self.hparams['merges_file']
            if self.hparams.get('max_len'):
                self.max_len = self.hparams['max_len']

        if not os.path.isfile(vocab_file):
            raise ValueError("Can't find a vocabulary file at path "
                             "'{}".format(vocab_file))

        if not os.path.isfile(merges_file):
            raise ValueError("Can't find a merges file at path "
                             "'{}".format(merges_file))

        with open(vocab_file) as fp:
            self.encoder = json.load(fp)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = self.hparams["errors"]  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_file, encoding='utf-8') as fp:
            bpe_data = fp.read().split('\n')[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_data]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache: Dict[str, str] = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for
        # capitalized versions of contractions
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?
            [^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def _map_text_to_token(self, text: str) -> List[str]:  # type: ignore
        r"""Tokenize a string. """
        bpe_tokens: List[str] = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(
                bpe_token for bpe_token in self._bpe(token).split(' '))
        return bpe_tokens

    def save_vocab(self, save_dir: str) -> Tuple[str, str]:
        r"""Save the tokenizer vocabulary and merge files to a directory."""
        if not os.path.isdir(save_dir):
            raise ValueError("Vocabulary path ({}) should be a "
                             "directory".format(save_dir))

        vocab_file = os.path.join(save_dir,
                                  self._VOCAB_FILE_NAMES['vocab_file'])
        merge_file = os.path.join(save_dir,
                                  self._VOCAB_FILE_NAMES['merges_file'])

        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write(u'#version: 0.2\n')
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(),
                                                  key=lambda kv: kv[1]):
                if index != token_index:
                    print("Saving vocabulary to {}: BPE merge indices are "
                          "not consecutive. Please check that the tokenizer "
                          "is not corrupted!".format(merge_file))
                    index = token_index
                writer.write(' '.join(bpe_tokens) + u'\n')
                index += 1

        return (vocab_file, merge_file)

    def _bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(
                pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: List[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 \
                        and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def _map_token_to_id(self, token: str) -> int:
        r"""Maps a token to an id using the vocabulary."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _map_id_to_token(self, index: int) -> str:
        r"""Maps an id to a token using the vocabulary."""
        token = self.decoder.get(index)
        assert isinstance(token, str)
        return token

    def map_token_to_text(self, tokens: List[str]) -> str:
        r"""Maps a sequence of tokens (string) in a single string."""
        text = ''.join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            'utf-8', errors=self.errors)
        return text

    def encode_text(  # type: ignore
            self,
            text: str,
            max_seq_length: Optional[int] = None,
            append_eos_token: bool = True) -> Tuple[List[int], int]:
        r"""Adds special tokens to a sequence and computes the corresponding
        sequence length for GPT2 specific tasks. The sequence will be truncated
        if its length is larger than ``max_seq_length``.

        A GPT2 sequence has the following format:
        `[bos_token]` X `[eos_token]` `[pad_token]`

        Args:
            text: Input text.
            max_seq_length: Maximum sequence length.
            append_eos_token: Whether to append ``eos_token`` after the
                sequence.

        Returns:
            A tuple of `(input_ids, seq_len)`, where

            - ``input_ids``: A list of input token ids with added
              special tokens.
            - ``seq_len``: The sequence length.
        """
        if max_seq_length is None:
            max_seq_length = self.max_len

        token_ids = self.map_text_to_id(text)
        assert isinstance(token_ids, list)

        bos_token_id = self._map_token_to_id(self.bos_token)
        eos_token_id = self._map_token_to_id(self.eos_token)
        pad_token_id = self._map_token_to_id(self.pad_token)

        if append_eos_token:
            input_ids = token_ids[:max_seq_length - 2]
            input_ids = [bos_token_id] + input_ids + [eos_token_id]
        else:
            input_ids = token_ids[:max_seq_length - 1]
            input_ids = [bos_token_id] + input_ids

        seq_len = len(input_ids)

        # Pad up to the maximum sequence length.
        input_ids = input_ids + [pad_token_id] * (max_seq_length - seq_len)

        assert len(input_ids) == max_seq_length

        return input_ids, seq_len

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        * The tokenizer is determined by the constructor argument
          :attr:`pretrained_model_name` if it's specified. In this case,
          `hparams` are ignored.
        * Otherwise, the tokenizer is determined by
          `hparams['pretrained_model_name']` if it's specified. All other
          configurations in `hparams` are ignored.
        * If the above two are `None`, the tokenizer is defined by the
          configurations in `hparams`.

        .. code-block:: python

            {
                "pretrained_model_name": "117M",
                "vocab_file": None,
                "merges_file": None,
                "max_len": 1024,
                "bos_token": "<|endoftext|>",
                "eos_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "errors": "replace",
            }

        Here:

        `"pretrained_model_name"`: str or None
            The name of the pre-trained GPT2 model.

        `"vocab_file"`: str or None
            The path to a vocabulary json file mapping tokens to ids.

        `"merges_file"`: str or None
            The path to a merges file.

        `"max_len"`: int
            The maximum sequence length that this model might ever be used with.

        `"bos_token"`: str
            Beginning of sentence token

        `"eos_token"`: str
            End of sentence token

        `"unk_token"`: str
            Unknown token

        `"pad_token"`: str
            Padding token

        `"errors"`: str
            Response when mapping tokens to text fails. The possible values are
            `ignore`, `replace`, and `strict`.

        `"name"`: str
            Name of the tokenizer.
        """
        return {
            'pretrained_model_name': '117M',
            'vocab_file': None,
            'merges_file': None,
            'max_len': 1024,
            'bos_token': '<|endoftext|>',
            'eos_token': '<|endoftext|>',
            'unk_token': '<|endoftext|>',
            'pad_token': '<|endoftext|>',
            'errors': 'replace',
            'name': 'gpt2_tokenizer',
            '@no_typecheck': ['pretrained_model_name'],
        }

    @classmethod
    def _transform_config(cls, pretrained_model_name: str,
                          cache_dir: str):
        r"""Returns the configuration of the pre-trained GPT2 tokenizer."""
        return {
            'vocab_file': None,
            'merges_file': None,
            'max_len': 1024,
            'bos_token': '<|endoftext|>',
            'eos_token': '<|endoftext|>',
            'unk_token': '<|endoftext|>',
            'pad_token': '<|endoftext|>',
            'errors': 'replace',
        }
