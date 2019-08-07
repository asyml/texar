"""Configurations of Transformer model
"""
import copy
import texar.tf as tx

random_seed = 1234
beam_width = 5
length_penalty = 0.6
hidden_dim = 512

emb = {
    'name': 'lookup_table',
    'dim': hidden_dim,
    'initializer': {
        'type': 'random_normal_initializer',
        'kwargs': {
            'mean': 0.0,
            'stddev': hidden_dim**-0.5,
        },
    }
}

position_embedder_hparams = {
    'dim': hidden_dim
}

encoder = {
    'dim': hidden_dim,
    'num_blocks': 6,
    'multihead_attention': {
        'num_heads': 8,
        'output_dim': hidden_dim
        # See documentation for more optional hyperparameters
    },
    'initializer': {
        'type': 'variance_scaling_initializer',
        'kwargs': {
            'scale': 1.0,
            'mode': 'fan_avg',
            'distribution': 'uniform',
        },
    },
    'poswise_feedforward': tx.modules.default_transformer_poswise_net_hparams(
        output_dim=hidden_dim)
}

decoder = copy.deepcopy(encoder)

loss_label_confidence = 0.9

opt = {
    'optimizer': {
        'type': 'AdamOptimizer',
        'kwargs': {
            'beta1': 0.9,
            'beta2': 0.997,
            'epsilon': 1e-9
        }
    }
}

lr = {
    'learning_rate_schedule': 'constant.linear_warmup.rsqrt_decay.rsqrt_depth',
    'lr_constant': 2 * (hidden_dim ** -0.5),
    'static_lr': 1e-3,
    'warmup_steps': 16000,
}
