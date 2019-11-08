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

import collections
import json
import os
import re
import warnings

from abc import ABC
from typing import Any, Dict

import tensorflow as tf
import numpy as np

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
      * ``gpt2-xl``: XL version of GPT-2, 1558M parameters.

    We provide the following GPT2 classes:

      * :class:`~texar.tf.modules.GPT2Encoder` for text encoding.
      * :class:`~texar.tf.modules.GPT2Decoder` for text generation and
        decoding.
      * :class:`~texar.tf.modules.GPT2Classifier` for text classification and
        sequence tagging.

    .. _`Language Models are Unsupervised Multitask Learners`:
        https://openai.com/blog/better-language-models/
    """
    _IS_DECODE = False
    _MODEL_NAME = "GPT2"
    _MODEL2URL = {
        'gpt2-small': [_GPT2_PATH + f"124M/{file}"
                       for file in _CHECKPOINT_FILES],
        'gpt2-medium': [_GPT2_PATH + f"355M/{file}"
                        for file in _CHECKPOINT_FILES],
        'gpt2-large': [_GPT2_PATH + f"774M/{file}"
                       for file in _CHECKPOINT_FILES],
        'gpt2-xl': [_GPT2_PATH + f"1558M/{file}"
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

    def _transform_config(self, pretrained_model_name: str,
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

        module_name = "decoder" if self._IS_DECODE else "encoder"
        configs.update({module_name: {
            "dim": hidden_dim,
            "num_blocks": config_gpt["n_layer"],
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
                        'factor': 1.0,
                        'mode': 'FAN_AVG',
                        'uniform': True
                },
            },
            "poswise_feedforward": {
                "layers": [
                    {
                        "type": "Dense",
                        "kwargs": {
                            'name': 'intermediate',
                            'activation': 'gelu',
                            "units": hidden_dim * 4,
                            "use_bias": True,
                        }
                    },
                    {
                        "type": "Dense",
                        "kwargs": {
                            'activation': None,
                            'name': 'output',
                            "units": hidden_dim,
                            "use_bias": True,
                        }
                    }
                ],
            },
        }})
        return configs

    def _init_from_checkpoint(self, pretrained_model_name, cache_dir,
                              scope_name, load_output_layer=True, **kwargs):
        r"""Initialize model parameters from weights stored in the pre-trained
        checkpoint.

        Args:
            pretrained_model_name (str): Name of the pre-trained model.
            cache_dir (str): Path to the cache directory.
            scope_name (str): Scope name of the model.
            load_output_layer (bool): If `False`, will not load weights of the
                output layer. Set this argument to `False` when loading weights
                into a GPT2 encoder. Defaults to `True`.
        """
        init_checkpoint = os.path.abspath(os.path.join(cache_dir,
                                                       'model.ckpt'))
        ckpt = tf.train.load_checkpoint(init_checkpoint)
        ckpt_params = {key: ckpt.get_tensor(key) for key in
                       ckpt.get_variable_to_shape_map().keys()}

        tvars = tf.trainable_variables()
        name_to_variable = collections.OrderedDict()
        for var in tvars:
            name = var.name
            m = re.match("^(.*):\\d+$", name)
            if m is not None:
                name = m.group(1)
            name_to_variable[name] = var

        if load_output_layer:
            global_tensor_map = {
                'model/wte': scope_name + '/word_embeddings/w',
                'model/wpe': scope_name + '/position_embeddings/w',
                'model/ln_f/b': scope_name + '/decoder/beta',
                'model/ln_f/g': scope_name + '/decoder/gamma',
            }

            layer_tensor_map = {
                "ln_1/b": scope_name + '/layer_{}/beta',
                "ln_1/g": scope_name + '/layer_{}/gamma',
                "ln_2/b": scope_name + '/layer_{}/past_poswise_ln/beta',
                "ln_2/g": scope_name + '/layer_{}/past_poswise_ln/gamma',
                "mlp/c_fc/b": scope_name + '/decoder/layer_{}'
                                           '/ffn/intermediate/bias',
                "mlp/c_fc/w": scope_name + '/decoder/layer_{}'
                                           '/ffn/intermediate/kernel',
                "mlp/c_proj/b": scope_name + '/decoder/layer_{}/ffn/output/'
                                             'bias',
                "mlp/c_proj/w": scope_name + '/decoder/layer_{}/ffn/output/'
                                             'kernel',
                "attn/c_attn/b": None,
                "attn/c_attn/w": None,
                "attn/c_proj/b": scope_name + '/decoder/layer_{}'
                                              '/self_attention/self/output/'
                                              'bias',
                "attn/c_proj/w": scope_name + '/decoder/layer_{}'
                                              '/self_attention/self/output/'
                                              'kernel',
            }
        else:
            global_tensor_map = {
                'model/wte': scope_name + '/word_embeddings/w',
                'model/wpe': scope_name + '/position_embeddings/w',
                'model/ln_f/b': scope_name + '/encoder/LayerNorm/beta',
                'model/ln_f/g': scope_name + '/encoder/LayerNorm/gamma',
            }

            layer_tensor_map = {
                "ln_1/b": scope_name + '/encoder/layer_{}/LayerNorm/beta',
                "ln_1/g": scope_name + '/encoder/layer_{}/LayerNorm/gamma',
                "ln_2/b": scope_name + '/encoder/layer_{}/output/'
                                       'LayerNorm/beta',
                "ln_2/g": scope_name + '/encoder/layer_{}/output/'
                                       'LayerNorm/gamma',
                "mlp/c_fc/b": scope_name + '/encoder/layer_{}'
                                           '/ffn/intermediate/bias',
                "mlp/c_fc/w": scope_name + '/encoder/layer_{}'
                                           '/ffn/intermediate/kernel',
                "mlp/c_proj/b": scope_name + '/encoder/layer_{}/ffn/output/'
                                             'bias',
                "mlp/c_proj/w": scope_name + '/encoder/layer_{}/ffn/output/'
                                             'kernel',
                "attn/c_attn/b": None,
                "attn/c_attn/w": None,
                "attn/c_proj/b": scope_name + '/encoder/layer_{}'
                                              '/attention/self/output/bias',
                "attn/c_proj/w": scope_name + '/encoder/layer_{}'
                                              '/attention/self/output/kernel',
            }

        for name, array in ckpt_params.items():
            if name in global_tensor_map:
                v_name = global_tensor_map[name]
                pointer = name_to_variable[v_name]
                pointer._initializer_op = tf.assign(pointer._variable, array)
            else:
                name_tmp = name.split("/")
                layer_no = name_tmp[1][1:]
                name = "/".join(name_tmp[2:])

                if name in layer_tensor_map:
                    if name == "attn/c_attn/b":
                        if load_output_layer:
                            K = name_to_variable[
                                scope_name + '/decoder/layer_' + layer_no +
                                '/self_attention/self/key/bias']
                            Q = name_to_variable[
                                scope_name + '/decoder/layer_' + layer_no +
                                '/self_attention/self/query/bias']
                            V = name_to_variable[
                                scope_name + '/decoder/layer_' + layer_no +
                                '/self_attention/self/value/bias']
                        else:
                            K = name_to_variable[
                                scope_name + '/encoder/layer_' + layer_no +
                                '/attention/self/key/bias']
                            Q = name_to_variable[
                                scope_name + '/encoder/layer_' + layer_no +
                                '/attention/self/query/bias']
                            V = name_to_variable[
                                scope_name + '/encoder/layer_' + layer_no +
                                '/attention/self/value/bias']

                        index_d = array.shape[-1] // 3

                        Q_w = array[:index_d]
                        K_w = array[index_d: 2 * index_d]
                        V_w = array[2 * index_d:]

                        K._initializer_op = tf.assign(K._variable, K_w)
                        Q._initializer_op = tf.assign(Q._variable, Q_w)
                        V._initializer_op = tf.assign(V._variable, V_w)
                    elif name == "attn/c_attn/w":
                        if load_output_layer:
                            K = name_to_variable[
                                scope_name + '/decoder/layer_' + layer_no +
                                '/self_attention/self/key/kernel']
                            Q = name_to_variable[
                                scope_name + '/decoder/layer_' + layer_no +
                                '/self_attention/self/query/kernel']
                            V = name_to_variable[
                                scope_name + '/decoder/layer_' + layer_no +
                                '/self_attention/self/value/kernel']
                        else:
                            K = name_to_variable[
                                scope_name + '/encoder/layer_' + layer_no +
                                '/attention/self/key/kernel']
                            Q = name_to_variable[
                                scope_name + '/encoder/layer_' + layer_no +
                                '/attention/self/query/kernel']
                            V = name_to_variable[
                                scope_name + '/encoder/layer_' + layer_no +
                                '/attention/self/value/kernel']

                        index_d = array.shape[-1] // 3

                        Q_w = np.transpose(array[0, :, :index_d])
                        K_w = np.transpose(array[0, :, index_d: 2 * index_d])
                        V_w = np.transpose(array[0, :, 2 * index_d:])

                        K._initializer_op = tf.assign(K._variable, K_w)
                        Q._initializer_op = tf.assign(Q._variable, Q_w)
                        V._initializer_op = tf.assign(V._variable, V_w)
                    elif (name == "attn/c_proj/w" or name == "mlp/c_fc/w" or
                          name == "mlp/c_proj/w"):
                        v_name = layer_tensor_map[name]
                        pointer = name_to_variable[v_name.format(layer_no)]
                        pointer._initializer_op = tf.assign(pointer._variable,
                                                            array[0])
                    else:
                        v_name = layer_tensor_map[name]
                        pointer = name_to_variable[v_name.format(layer_no)]
                        pointer._initializer_op = tf.assign(pointer._variable,
                                                            array)
