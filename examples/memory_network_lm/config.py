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

n_hops = 7
dim = 150
relu_dim = dim // 2
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

embed = {
    "embedding": {
        "dim": dim,
    },
    "temporal_embedding": {
        "dim": dim,
    }
}

memnet = {
    "n_hops": n_hops,
    "relu_dim": relu_dim,
    "memory_size": memory_size,
    "A": embed,
    "C": embed,
}
