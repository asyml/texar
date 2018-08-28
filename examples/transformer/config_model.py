# -*- coding: utf-8 -*-
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
