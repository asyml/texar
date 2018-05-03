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
"""PTB LM large size config.
"""

# pylint: disable=invalid-name, too-few-public-methods, missing-docstring

init_scale = 0.04
num_epochs = 55
hidden_size = 1500
keep_prob = 0.35
batch_size = 20
num_steps = 35

cell = {
    "type": "LSTMBlockCell",
    "kwargs": {
        "num_units": hidden_size,
        "forget_bias": 0.
    },
    "dropout": {"output_keep_prob": keep_prob},
    "num_layers": 2
}
emb = {
    "dim": hidden_size
}
opt = {
    "optimizer": {
        "type": "GradientDescentOptimizer",
        "kwargs": {"learning_rate": 1.0}
    },
    "gradient_clip": {
        "type": "clip_by_global_norm",
        "kwargs": {"clip_norm": 10.}
    },
    "learning_rate_decay": {
        "type": "exponential_decay",
        "kwargs": {
            "decay_steps": 1,
            "decay_rate": 1. / 1.15,
            "staircase": True
        },
        "start_decay_step": 14
    }
}
