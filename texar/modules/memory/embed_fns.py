# Copyright 2018 The Texar Authors. All Rights Reserved.
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
Some embed_fn s used in :class:`~texar.modules.memory.MemNetBase` and its
subclasses.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.modules.embedders import WordEmbedder, PositionEmbedder

__all__ = [
    'default_embed_fn',
    'get_default_embed_fn_hparams',
]

def get_default_embed_fn_hparams():
    """Returns a dictionary of hyperparameters with default hparams for
    :func:`~texar.modules.memory.default_embed_fn`

    Returns:
        .. code-block:: python

            {
                "memory_size": 100,
                "embedding": {
                    "name": "embedding",
                    "dim": 100,
                    "initializer": None, # use default initializer
                    "dropout_rate": 0
                }
                "temporal_embedding": {
                    "name": "temporal_embedding",
                    "dim": 100,
                    "initializer": None, # use default initializer
                    "dropout_rate": 0
                }
            }
    """
    return {
        "memory_size": 100,
        "embedding": {
            "name": "embedding",
            "dim": 100,
            "initializer": None,
            "dropout_rate": 0
        },
        "temporal_embedding": {
            "name": "temporal_embedding",
            "dim": 100,
            "dropout_rate": 0
        }
    }

def default_embed_fn(memory, soft_memory, vocab_size, hparams):
    """Default embed function for A, C or B operation.

    Args:
        memory: Memory indices used for embedding lookup.
        vocab_size(int): Size of vocabulary used for embedding.
        hparams(HParams or dict): Hyperparameters of this function.
            See :func:`~texar.modules.memory.get_default_embed_fn_hparams`.

    Returns:
        Result of the memory operation.
        In this case, :attr:`embedded_memory + temporal_embedding`.
    """
    if memory is None and soft_memory is None:
        raise ValueError("Either `memory` or `soft_memory` is required.")
    if memory is not None and soft_memory is not None:
        raise ValueError(
            "Must not specify `memory` and `soft_memory` at the same time.")

    memory_size = hparams["memory_size"]

    embedding = WordEmbedder(
        vocab_size=vocab_size,
        hparams=hparams["embedding"]
    )
    embedded_memory = embedding(ids=memory, soft_ids=soft_memory)

    # temporal embedding
    temporal_embedding = PositionEmbedder(
        position_size=memory_size,
        hparams=hparams["temporal_embedding"]
    )
    temporal_embedded = temporal_embedding(
        sequence_length=tf.constant([memory_size]))

    return tf.add(embedded_memory, temporal_embedded)
