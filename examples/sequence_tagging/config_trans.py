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
load_glove = False
embed_dim = 100
hidden_size = 100
tag_space = 128
batch_size = 16
dropout = 0.1
encoder = 'transformer'
encoder_hparams = {
    'num_units': hidden_size,
    'multiply_embedding_mode': "sqrt_depth",
    'embedding_dropout': dropout,
    'attention_dropout': dropout,
    'residual_dropout': dropout,
    'sinusoid': True,
    'num_blocks': 6,
    'num_heads': 4,
    'zero_pad': False,
    'bos_pad': False,
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
                    'rate': dropout,
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

opt = {
    "optimizer": {
        "type": "AdamOptimizer",
    },
    #"learning_rate_decay": {
    #    "type": "inverse_time_decay",
    #    "kwargs": {
    #        "decay_steps": 1,
    #        "decay_rate": 0.05,
    #        "staircase": True
    #    },
    #    "start_decay_step": 1
    #}
}
