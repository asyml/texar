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

# pylint: disable=invalid-name, too-few-public-methods, missing-docstring

n_hops = 10
dim = 360
reludim = dim // 2
batch_size = 128
num_epochs = 200
memory_size = 200
initialize_stddev = 0.05
query_constant = 0.1
learning_rate_anneal_factor = 1.5
terminating_learning_rate = 1e-5

opt = {
    "optimizer": {
        "type": "GradientDescentOptimizer",
        "kwargs": {"learning_rate": 0.01}
    },
    "gradient_clip": {
        "type": "clip_by_global_norm",
        "kwargs": {"clip_norm": 50.}
    },
}

embedder = {
    "memory_size": memory_size,
    "word_embedder": {
        "dim": dim,
        "dropout_rate": 0.2
    },
    "temporal_embedding": {
        "dim": dim,
        "dropout_rate": 0.2
    }
}

memnet = {
    "n_hops": n_hops,
    "dim": dim,
    "reludim": reludim,
    "memory_size": memory_size,
    "need_H": True,
    "final_matrix": {
        "dim": dim,
        "dropout_rate": 0.2
    },
    "A": embedder,
    "C": embedder,
    "dropout_rate": 0.2,
    "variational": True
}
