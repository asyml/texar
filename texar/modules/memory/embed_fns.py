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

# pylint: disable=invalid-name, too-many-arguments

__all__ = [
    'default_memnet_embed_fn_hparams',
]

def default_memnet_embed_fn_hparams():
    """Returns a dictionary of hyperparameters with default hparams for
    :func:`~texar.modules.memory.default_embed_fn`

    .. code-block:: python

        {
            "embedding": {
                "dim": 100
            },
            "temporal_embedding": {
                "dim": 100
            },
            "combine_mode": "add"
        }

    Here:

    "embedding" : dict, optional
        Hyperparameters for embedding operations. See
        :meth:`~texar.modules.WordEmbedder.default_hparams` of
        :class:`~texar.modules.WordEmbedder` for details. If `None`, the
        default hyperparameters are used.

    "temporal_embedding" : dict, optional
        Hyperparameters for temporal embedding operations. See
        :meth:`~texar.modules.PositionEmbedder.default_hparams` of
        :class:`~texar.modules.PositionEmbedder` for details. If `None`, the
        default hyperparameters are used.

    "combine_mode" : str
        Either **'add'** or **'concat'**. If 'add', memory
        embedding and temporal embedding are added up. In this case the two
        embedders must have the same dimension. If 'concat', the two
        embeddings are concated.
    """
    return {
        "embedding": {
            "dim": 100
        },
        "temporal_embedding": {
            "dim": 100
        },
        "combine_mode": "add"
    }

