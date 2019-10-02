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
Utils of GPT2 Modules.
"""

import json
import os
import sys
import warnings
from abc import ABC
from typing import Any, Dict

import torch

from texar.tf.modules.pretrained.pretrained_base import PretrainedMixin

__all__ = [
    "PretrainedGPT2Mixin",
]

_GPT2_PATH = "https://storage.googleapis.com/gpt-2/models/"
_CHECKPOINT_FILES = [
    "checkpoint", "encoder.json", "hparams.json", "vocab.bpe",
    "model.ckpt.data-00000-of-00001", "model.ckpt.index", "model.ckpt.meta"]


class PretrainedGPT2Mixin(PretrainedMixin, ABC):
    r"""A mixin class to support loading pre-trained checkpoints for modules
    that implement the GPT2 model.

    The GPT2 model was proposed in
    `Language Models are Unsupervised Multitask Learners`_
    by `Radford et al.` from OpenAI. It is a unidirectional Transformer model
    pre-trained using the vanilla language modeling objective on a large corpus.

    The available GPT2 models are as follows:

      * ``gpt2-small``: Small version of GPT-2, 124M parameters.
      * ``gpt2-medium``: Medium version of GPT-2, 355M parameters.
      * ``gpt2-large``: Large version of GPT-2, 774M parameters.

    We provide the following GPT2 classes:

      * :class:`~texar.tf.modules.GPT2Encoder` for text encoding.
      * :class:`~texar.tf.modules.GPT2Decoder` for text generation and
        decoding.
      * :class:`~texar.tf.modules.GPT2Classifier` for text classification and
        sequence tagging.

    .. _`Language Models are Unsupervised Multitask Learners`:
        https://openai.com/blog/better-language-models/
    """
    _MODEL_NAME = "GPT2"
    _MODEL2URL = {
        'gpt2-small': [_GPT2_PATH + f"124M/{file}"
                       for file in _CHECKPOINT_FILES],
        'gpt2-medium': [_GPT2_PATH + f"355M/{file}"
                        for file in _CHECKPOINT_FILES],
        'gpt2-large': [_GPT2_PATH + f"774M/{file}"
                       for file in _CHECKPOINT_FILES],
    }

    # Raise warning for the deprecated pre-trained model names
    class MyDict(dict):
        def __contains__(self, key):
            if key == '117M':
                warnings.warn("Pre-trained model name '117M' is deprecated, "
                              "use 'gpt2-small' instead.", UserWarning)
                return True
            elif key == '345M':
                warnings.warn("Pre-trained model name '345M' is deprecated, "
                              "use 'gpt2-medium' instead.", UserWarning)
                return True
            else:
                return super().__contains__(key)

    _DEPRECATED_MODEL2URL = {
        '117M': [_GPT2_PATH + f"124M/{file}" for file in _CHECKPOINT_FILES],
        '345M': [_GPT2_PATH + f"355M/{file}" for file in _CHECKPOINT_FILES],
    }
    _MODEL2URL.update(_DEPRECATED_MODEL2URL)
    _MODEL2URL = MyDict(_MODEL2URL)  # type: ignore

    @classmethod
    def _transform_config(cls, pretrained_model_name: str,
                          cache_dir: str) -> Dict[str, Any]:
        info = list(os.walk(cache_dir))
        root, _, files = info[0]
        config_path = None
        for file in files:
            if file.endswith('hparams.json'):
                config_path = os.path.join(root, file)
        if config_path is None:
            raise ValueError(f"Cannot find the config file in {cache_dir}")

        with open(config_path) as f:
            config_gpt = json.loads(f.read())

        hidden_dim = config_gpt["n_embd"]
        configs = {
            "vocab_size": config_gpt["n_vocab"],
            "context_size": config_gpt["n_ctx"],
            "embedding_size": config_gpt["n_embd"], "embed": {
                "dim": hidden_dim,
            },
            "position_size": config_gpt["n_ctx"],
            "position_embed": {
                "dim": hidden_dim
            }
        }
        configs.update({
            "dim": hidden_dim,
            "num_blocks": config_gpt["n_layer"],
            "use_gpt_config": True,
            "embedding_dropout": 0,
            "residual_dropout": 0,
            "multihead_attention": {
                "use_bias": True,
                "num_units": hidden_dim,
                "num_heads": config_gpt["n_head"],
                "output_dim": hidden_dim,
            },
            "initializer": {
                "type": "variance_scaling_initializer",
                "kwargs": {
                    "factor": 1.0,
                    "mode": "FAN_AVG",
                    "uniform": True,
                },
            },
            "poswise_feedforward": {
                "layers": [
                    {
                        "type": "Linear",
                        "kwargs": {
                            "in_features": hidden_dim,
                            "out_features": hidden_dim * 4,
                            "bias": True,
                        }
                    },
                    {
                        "type": "GPTGELU",
                        "kwargs": {}
                    },
                    {
                        "type": "Linear",
                        "kwargs": {
                            "in_features": hidden_dim * 4,
                            "out_features": hidden_dim,
                            "bias": True,
                        }
                    }
                ],
                "name": "ffn",
            },
        })
        return configs

