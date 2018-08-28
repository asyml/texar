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
import copy
random_seed = 1234
beam_width = 5
alpha = 0.6
max_decode_len = 256
hidden_dim = 512
word_embedding_hparams = {
    'name': 'lookup_table',
    'dim': hidden_dim,
    'initializer': {
        'type': 'random_normal_initializer',
        'kwargs': {
            'mean':0.0,
            'stddev':hidden_dim**-0.5,
        },
    }
}
encoder_hparams = {
    'position_embedder_hparams': {
        'name': 'sinusoids',
    },
    'dim': hidden_dim,
    'initializer': {
        'type': 'variance_scaling_initializer',
        'kwargs': {
            'scale':1.0,
            'mode':'fan_avg',
            'distribution':'uniform',
        },
    },
}
decoder_hparams = copy.deepcopy(encoder_hparams)
decoder_hparams['max_decoding_length'] = max_decode_len
loss_hparams = {
    'label_confidence': 0.9,
}

opt_hparams = {
    'learning_rate_schedule':
        'constant.linear_warmup.rsqrt_decay.rsqrt_depth',
    'lr_constant': 2 * (hidden_dim ** -0.5),
    'static_lr': 1e-3,
    'warmup_steps': 16000,
    'Adam_beta1':0.9,
    'Adam_beta2':0.997,
    'Adam_epsilon':1e-9,
}
