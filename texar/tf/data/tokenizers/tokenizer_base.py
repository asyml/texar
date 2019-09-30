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
Base class for all tokenizers.

The code structure adapted from:
    `https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/tokenization_utils.py`
"""

from typing import Any, Dict, List, Optional, Tuple, overload

import os
import json

from texar.tf.module_base import ModuleBase

__all__ = [
    "TokenizerBase",
]

SPECIAL_TOKENS_MAP_FILE = 'special_tokens_map.json'
ADDED_TOKENS_FILE = 'added_tokens.json'
CONFIG_FILE = 'config.json'


class TokenizerBase(ModuleBase):
    r"""Base class inherited by all tokenizer classes. This class
    handles downloading and loading pre-trained tokenizer and adding tokens to
    the vocabulary.

    Derived class can set up a few special tokens to be used in common scripts
    and internals: :attr:`bos_token`, :attr:`eos_token`, :attr:`unk_token`,
    :attr:`sep_token`, :attr:`pad_token`, :attr:`cls_token`,
    :attr:`mask_token`, and :attr:`additional_special_tokens`.

    We defined an :attr:`added_tokens_encoder` to add new tokens to the
    vocabulary without having to handle the specific vocabulary augmentation
    methods of the various underlying dictionary structures (`BPE`,
    `sentencepiece` ...).
    """

    _IS_PRETRAINED: bool
    _MAX_INPUT_SIZE: Dict[str, Optional[int]]
    _VOCAB_FILE_NAMES: Dict[str, str]
    _SPECIAL_TOKENS_ATTRIBUTES = ["bos_token", "eos_token", "unk_token",
                                  "sep_token", "pad_token", "cls_token",
                                  "mask_token", "additional_special_tokens"]

    def __init__(self, hparams):
        super().__init__(hparams=hparams)

        self.config = None

        self.bos_token = None
        self.eos_token = None
        self.unk_token = None
        self.sep_token = None
        self.pad_token = None
        self.cls_token = None
        self.mask_token = None
        self.additional_special_tokens = []

        self.max_len = int(1e12)
        self.added_tokens_encoder = {}
        self.added_tokens_decoder = {}

        for key, value in self.hparams.items():
            if key in self._SPECIAL_TOKENS_ATTRIBUTES:
                if key == 'additional_special_tokens':
                    assert isinstance(value, (list, tuple)) and \
                           all(isinstance(v, str) for v in value)
                else:
                    assert isinstance(value, str)
                setattr(self, key, value)

    @classmethod
    def load(cls, pretrained_model_path: str, configs: Optional[Dict] = None):
        r"""Instantiate a tokenizer from the vocabulary files or the saved
        tokenizer files.

        Args:
            pretrained_model_path: The path to a vocabulary file or a folder
                that contains the saved pre-trained tokenizer files.
            configs: Tokenizer configurations. You can overwrite the original
                tokenizer configurations saved in the configuration file
                by this dictionary.

        Returns:
            A tokenizer instance.
        """
        vocab_files = {}
        # Look for the tokenizer main vocabulary files
        for file_id, file_name in cls._VOCAB_FILE_NAMES.items():
            full_file_name: Optional[str]
            if os.path.isdir(pretrained_model_path):
                # If a directory is provided we look for the standard file name
                full_file_name = os.path.join(pretrained_model_path, file_name)
            else:
                # If a path to a file is provided we use it (will only work
                # for non-BPE tokenizer using a single vocabulary file)
                full_file_name = pretrained_model_path
            if not os.path.exists(full_file_name):
                print("Can't find file {}. We won't load it.".format(
                    full_file_name))
                full_file_name = None
            vocab_files[file_id] = full_file_name

        # Look for the additional tokens files
        all_vocab_files_names = {
            'added_tokens_file': ADDED_TOKENS_FILE,
            'special_tokens_map_file': SPECIAL_TOKENS_MAP_FILE,
            'config_file': CONFIG_FILE}

        # If a path to a file was provided, get the parent directory
        saved_directory = pretrained_model_path
        if os.path.exists(saved_directory) and not os.path.isdir(
                saved_directory):
            saved_directory = os.path.dirname(saved_directory)

        for file_id, file_name in all_vocab_files_names.items():
            full_file_name = os.path.join(saved_directory, file_name)
            if not os.path.exists(full_file_name):
                print("Can't find file {}. We won't load it.".format(
                    full_file_name))
                full_file_name = None
            vocab_files[file_id] = full_file_name

        if all(full_file_name is None for full_file_name in
               vocab_files.values()):
            raise ValueError("Can't find tokenizer files in {}.".format(
                saved_directory))

        kwargs: Dict[str, Any]
        if cls._IS_PRETRAINED:
            kwargs = {'pretrained_model_name': None}
        else:
            kwargs = {}

        added_tokens_file = vocab_files.pop('added_tokens_file', None)
        special_tokens_map_file = vocab_files.pop(
            'special_tokens_map_file', None)
        tokenizer_config_file = vocab_files.pop('config_file', None)

        for args_name, file_path in vocab_files.items():
            if args_name not in kwargs:
                kwargs[args_name] = file_path

        if special_tokens_map_file is not None:
            with open(special_tokens_map_file, encoding="utf-8") as f:
                special_tokens_map = json.load(f)
            for key, value in special_tokens_map.items():
                if key not in kwargs:
                    kwargs[key] = value

        if tokenizer_config_file is not None:
            with open(tokenizer_config_file, encoding="utf-8") as f:
                tokenizer_config = json.load(f)
            for key, value in tokenizer_config.items():
                if key not in kwargs:
                    kwargs[key] = value

        if configs is not None:
            for key, value in configs.items():
                kwargs[key] = value

        tokenizer = cls(hparams=kwargs)

        # Add supplementary tokens.
        if added_tokens_file is not None:
            with open(added_tokens_file, encoding="utf-8") as f:
                added_tok_encoder = json.load(f)
            added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
            tokenizer.added_tokens_encoder.update(added_tok_encoder)
            tokenizer.added_tokens_decoder.update(added_tok_decoder)

        return tokenizer

    def save(self, save_dir: str) -> Tuple[str]:
        r"""Save the tokenizer vocabulary files (with added tokens), tokenizer
        configuration file and a dictionary mapping special token class
        attributes (:attr:`cls_token`, :attr:`unk_token`, ...) to their values
        (`<unk>`, `<cls>`, ...) to a directory, so that it can be re-loaded
        using the :meth:`~load`.

        Args:
            save_dir: The path to a folder in which the tokenizer files
                will be saved.

        Return:
            The paths to the vocabulary file, added token file, special token
            mapping file, and the configuration file.
        """
        if not os.path.isdir(save_dir):
            raise ValueError("Saving directory ({}) should be a "
                             "directory".format(save_dir))

        special_tokens_map_file = os.path.join(save_dir,
                                               SPECIAL_TOKENS_MAP_FILE)
        added_tokens_file = os.path.join(save_dir, ADDED_TOKENS_FILE)
        config_file = os.path.join(save_dir, CONFIG_FILE)

        with open(special_tokens_map_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.special_tokens_map, ensure_ascii=False))

        with open(added_tokens_file, 'w', encoding='utf-8') as f:
            if self.added_tokens_encoder:
                out_str = json.dumps(self.added_tokens_encoder,
                                     ensure_ascii=False)
            else:
                out_str = u"{}"
            f.write(out_str)

        with open(config_file, 'w', encoding='utf-8') as f:
            if self.config:
                out_str = json.dumps(self.config, ensure_ascii=False)
            else:
                out_str = u"{}"
            f.write(out_str)

        vocab_files = self.save_vocab(save_dir)
        return vocab_files + (special_tokens_map_file, added_tokens_file,
                              config_file)

    def save_vocab(self, save_dir):
        r"""Save the tokenizer vocabulary to a directory. This method does not
        save added tokens, special token mappings, and the configuration file.

        Please use :meth:`~save` to save the full tokenizer state so
        that it can be reloaded using :meth:`~load`.
        """
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.vocab_size + len(self.added_tokens_encoder)

    def add_tokens(self, new_tokens: List[Optional[str]]) -> int:
        r"""Add a list of new tokens to the tokenizer class. If the new tokens
        are not in the vocabulary, they are added to the
        :attr:`added_tokens_encoder` with indices starting from the last index
        of the current vocabulary.

        Args:
            new_tokens: A list of new tokens.

        Returns:
            Number of tokens added to the vocabulary which can be used to
            correspondingly increase the size of the associated model embedding
            matrices.
        """
        if not new_tokens:
            return 0

        to_add_tokens = []
        for token in new_tokens:
            assert isinstance(token, str)
            if token != self.unk_token and \
                    (self.map_token_to_id(token) ==
                     self.map_token_to_id(self.unk_token)):
                to_add_tokens.append(token)

        added_tok_encoder = dict((tok, len(self) + i) for i, tok in
                                 enumerate(to_add_tokens))
        added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
        self.added_tokens_encoder.update(added_tok_encoder)
        self.added_tokens_decoder.update(added_tok_decoder)

        return len(to_add_tokens)

    def add_special_tokens(self, special_tokens_dict: Dict[str, str]) -> int:
        r"""Add a dictionary of special tokens to the encoder and link them to
        class attributes. If the special tokens are not in the vocabulary, they
        are added to it and indexed starting from the last index of the
        current vocabulary.

        Args:
            special_tokens_dict: A dictionary mapping special token class
                attributes (:attr:`cls_token`, :attr:`unk_token`, ...) to their
                values (`<unk>`, `<cls>`, ...).

        Returns:
            Number of tokens added to the vocabulary which can be used to
            correspondingly increase the size of the associated model embedding
            matrices.
        """
        if not special_tokens_dict:
            return 0

        added_tokens = 0
        for key, value in special_tokens_dict.items():
            assert key in self._SPECIAL_TOKENS_ATTRIBUTES
            if key == 'additional_special_tokens':
                assert isinstance(value, (list, tuple)) and all(
                    isinstance(t, str) for t in value)
                added_tokens += self.add_tokens(value)
            else:
                assert isinstance(value, str)
                added_tokens += self.add_tokens([value])
            setattr(self, key, value)

        return added_tokens

    def map_text_to_token(self, text: Optional[str],
                          **kwargs) -> List[str]:
        r"""Maps a string to a sequence of tokens (string), using the
        tokenizer. Split in words for word-based vocabulary or sub-words for
        sub-word-based vocabularies (`BPE`/`SentencePiece`/`WordPiece`).
        This function also takes care of the added tokens.

        Args:
            text: A input string.

        Return:
            A list of tokens.
        """

        def split_on_tokens(tok_list, string):
            if not string:
                return []
            if not tok_list:
                return self._map_text_to_token(string, **kwargs)
            tok = tok_list[0]
            split_text = string.split(tok)
            return sum((split_on_tokens(tok_list[1:], sub_text.strip()) + [tok]
                        for sub_text in split_text), [])[:-1]

        added_tokens = list(
            self.added_tokens_encoder.keys()) + self.all_special_tokens
        tokenized_text = split_on_tokens(added_tokens, text)
        return tokenized_text

    def _map_text_to_token(self, text: str, **kwargs) -> List[str]:
        r"""Maps a string to a sequence of tokens (string), using the
        tokenizer. Split in words for word-based vocabulary or sub-words for
        sub-word-based vocabularies (`BPE`/`SentencePiece`/`WordPiece`).
        This function does not take care of the added tokens.
        """
        raise NotImplementedError

    # TODO: Remove these once pylint supports function stubs.
    # pylint: disable=unused-argument,function-redefined

    @overload
    def map_token_to_id(self, tokens: str) -> int:
        ...

    @overload
    def map_token_to_id(self, tokens: List[str]) -> List[int]:
        ...

    def map_token_to_id(self, tokens):
        r"""Maps a single token or a sequence of tokens to a integer id
        (resp.) a sequence of ids, using the vocabulary.

        Args:
            tokens: A single token or a list of tokens.

        Returns:
            A single token id or a list of token ids.
        """
        if isinstance(tokens, str):
            return self._map_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._map_token_to_id_with_added_voc(token))
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified "
                "maximum sequence length for this model ({} > {}). Running "
                "this sequence through the model will result in indexing "
                "errors".format(len(ids), self.max_len))
        return ids

    # pylint: enable=unused-argument,function-redefined

    def _map_token_to_id_with_added_voc(self, token: str) -> int:
        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._map_token_to_id(token)

    def _map_token_to_id(self, token: str) -> int:
        raise NotImplementedError

    def map_text_to_id(self, text: str) -> List[int]:
        r"""Maps a string to a sequence of ids (integer), using the
        tokenizer and vocabulary. Same as
        `self.map_token_to_id(self.map_text_to_token(text))`.

        Args:
            text: A input string.

        Returns:
            A single token id or a list of token ids.
        """
        return self.map_token_to_id(self.map_text_to_token(text))

    # TODO: Remove these once pylint supports function stubs.
    # pylint: disable=unused-argument,function-redefined

    @overload
    def map_id_to_token(self, token_ids: int,
                        skip_special_tokens: bool = False) -> str:
        ...

    @overload
    def map_id_to_token(self, token_ids: List[int],
                        skip_special_tokens: bool = False) -> List[str]:
        ...

    def map_id_to_token(self, token_ids, skip_special_tokens=False):
        r"""Maps a single id or a sequence of ids to a token (resp.) a
        sequence of tokens, using the vocabulary and added tokens.

        Args:
            token_ids: A single token id or a list of token ids.
            skip_special_tokens: Whether to skip the special tokens.

        Returns:
            A single token or a list of tokens.
        """
        if isinstance(token_ids, int):
            if token_ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[token_ids]
            else:
                return self._map_id_to_token(token_ids)
        tokens = []
        for index in token_ids:
            if index in self.all_special_ids and skip_special_tokens:
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._map_id_to_token(index))
        return tokens

    # pylint: enable=unused-argument,function-redefined

    def _map_id_to_token(self, token_id: int) -> str:
        raise NotImplementedError

    def map_token_to_text(self, tokens: List[str]) -> str:
        r"""Maps a sequence of tokens (string) in a single string.
        The most simple way to do it is :python:`' '.join(tokens)`, but we
        often want to remove sub-word tokenization artifacts at the same time.
        """
        raise NotImplementedError

    def map_id_to_text(self, token_ids: List[int],
                       skip_special_tokens: bool = False,
                       clean_up_tokenization_spaces: bool = True) -> str:
        r"""Maps a sequence of ids (integer) to a string, using the
        tokenizer and vocabulary with options to remove special tokens and
        clean up tokenization spaces.

        Args:
            token_ids: A list of token ids.
            skip_special_tokens: Whether to skip the special tokens.
            clean_up_tokenization_spaces: Whether to clean up a list of simple
                English tokenization artifacts like spaces before punctuations
                and abbreviated forms.
        """
        filtered_tokens = self.map_id_to_token(
            token_ids, skip_special_tokens=skip_special_tokens)
        text = self.map_token_to_text(filtered_tokens)
        if clean_up_tokenization_spaces:
            text = self.clean_up_tokenization(text)
        return text

    def encode_text(self,
                    text_a: str,
                    text_b: Optional[str] = None,
                    max_seq_length: Optional[int] = None):
        r"""Adds special tokens to a sequence or sequence pair and computes
        other information such as segment ids, input mask, and sequence length
        for specific tasks.
        """
        raise NotImplementedError

    @property
    def special_tokens_map(self) -> Dict[str, str]:
        r"""A dictionary mapping special token class attributes
        (:attr:`cls_token`, :attr:`unk_token`, ...) to their values
        (`<unk>`, `<cls>`, ...)
        """
        set_attr = {}
        for attr in self._SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens(self) -> List[str]:
        r"""List all the special tokens (`<unk>`, `<cls>`, ...) mapped to class
        attributes (:attr:`cls_token`, :attr:`unk_token`, ...).
        """
        all_toks: List[str] = []
        set_attr = self.special_tokens_map
        for attr_value in set_attr.values():
            all_toks = all_toks + (
                attr_value if isinstance(attr_value, (list, tuple)) else [
                    attr_value])
        all_toks = list(set(all_toks))
        return all_toks

    @property
    def all_special_ids(self) -> List[int]:
        r"""List the vocabulary indices of the special tokens
        (`<unk>`, `<cls>`, ...) mapped to class attributes
        (:attr:`cls_token`, :attr:`unk_token`, ...).
        """
        all_toks = self.all_special_tokens
        all_ids: List[int] = [self.map_token_to_id(t) for t in all_toks]
        return all_ids

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        r"""Clean up a list of simple English tokenization artifacts like
        spaces before punctuations and abbreviated forms.
        """
        out_string = out_string.replace(' .', '.').replace(' ?', '?'). \
            replace(' !', '!').replace(' ,', ',').replace(" ' ", "'"). \
            replace(" n't", "n't").replace(" 'm", "'m"). \
            replace(" do not", " don't").replace(" 's", "'s"). \
            replace(" 've", "'ve").replace(" 're", "'re")
        return out_string
