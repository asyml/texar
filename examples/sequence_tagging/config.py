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
hidden_size = 256
tag_space = 128
keep_prob = 0.5
batch_size = 16
encoder = None
encoder_hparams = {
    'multiply_embedding_mode': "sqrt_depth",
    'embedding_dropout': 0.1,
    'attention_dropout': 0.1,
    'residual_dropout': 0.1,
    'sinusoid': True,
    'num_blocks': 6,
    'num_heads': 8,
    'zero_pad': 0,
    'bos_pad': 0,
    'initializer': {
        'type': 'variance_scaling_initializer',
        'kwargs': {
            'scale':1.0,
            'mode':'fan_avg',
            'distribution':'uniform',
        },
    },
    'poswise_feedforward': {
        'name':'ffn',
        'layers':[
            {
                'type':'Dense',
                'kwargs': {
                    'name':'conv1',
                    'units':hidden_size*4,
                    'activation':'relu',
                    'use_bias':True,
                }
            },
            {
                'type':'Dropout',
                'kwargs': {
                    'rate': 0.1,
                }
            },
            {
                'type':'Dense',
                'kwargs': {
                    'name':'conv2',
                    'units':hidden_size,
                    'use_bias':True,
                    }
            }
        ],
    },
}

cell = {
    "type": "LSTMBlockCell",
    "kwargs": {
        "num_units": hidden_size,
        "forget_bias": 0.
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
    "gradient_clip": {
        "type": "clip_by_global_norm",
        "kwargs": {"clip_norm": 5.}
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
