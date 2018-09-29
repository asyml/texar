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
"""NER config.
"""

# pylint: disable=invalid-name, too-few-public-methods, missing-docstring

num_epochs = 200
char_dim = 30
embed_dim = 100
hidden_size = 256
tag_space = 128
keep_prob = 0.5
batch_size = 16
encoder = None
load_glove = True

emb = {
    "name": "embedding",
    "dim": embed_dim,
    "dropout_rate": 0.33,
    "dropout_strategy": 'item'
}

char_emb = {
    "name": "char_embedding",
    "dim": char_dim
}

conv = {
    "filters": 30,
    "kernel_size": [3],
    "conv_activation": "tanh",
    "num_dense_layers": 0,
    "dropout_rate": 0.
}

cell = {
    "type": "LSTMCell",
    "kwargs": {
        "num_units": hidden_size,
        "forget_bias": 1.
    },
    "dropout": {"output_keep_prob": keep_prob},
    "num_layers": 1
}
opt = {
    "optimizer": {
        "type": "MomentumOptimizer",
        "kwargs": {"learning_rate": 0.1,
                   "momentum": 0.9,
                   "use_nesterov": True}
    },
    "learning_rate_decay": {
        "type": "inverse_time_decay",
        "kwargs": {
            "decay_steps": 1,
            "decay_rate": 0.05,
            "staircase": True
        },
        "start_decay_step": 1
    }
}
