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

"""VAE config.
"""

# pylint: disable=invalid-name, too-few-public-methods, missing-docstring

num_epochs = 30
hidden_size = 256
keep_prob = 0.5
batch_size = 32
emb_size = 300

latent_dims = 16

cell_hparams = {
    "type": "LSTMBlockCell",
    "kwargs": {
        "num_units": hidden_size,
        "forget_bias": 0.
    },
    "dropout": {"output_keep_prob": keep_prob},
    "num_layers": 1
}

emb_hparams = {
    "dim": emb_size
}


# KL annealing
# kl_weight = 1.0 / (1 + np.exp(-k*(step-x0)))
anneal_hparams = {
        "x0": 2500,
        "k": 0.0025
}

train_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "seed": 123,
    "dataset": {
        "files": 'data/ptb/ptb.train.txt',
        "vocab_file": 'data/ptb/vocab.txt'
    }
}

val_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "seed": 123,
    "dataset": {
        "files": 'data/ptb/ptb.val.txt',
        "vocab_file": 'data/ptb/vocab.txt'
    }
}

test_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "dataset": {
        "files": 'data/ptb/ptb.test.txt',
        "vocab_file": 'data/ptb/vocab.txt'
    }
}

opt_hparams = {
    "optimizer": {
        "type": "GradientDescentOptimizer",
        "kwargs": {"learning_rate": 1.0}
    },
    "gradient_clip": {
        "type": "clip_by_global_norm",
        "kwargs": {"clip_norm": 5.}
    },
    "learning_rate_decay": {
        "type": "exponential_decay",
        "kwargs": {
            "decay_steps": 1,
            "decay_rate": 0.5,
            "staircase": True
        },
        "start_decay_step": 3
    }
}
