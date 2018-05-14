# pylint: disable=invalid-name, too-few-public-methods, missing-docstring
num_epochs = 1000
embed_dim = 256
hidden_size = 256
train_batch_size = 64
valid_batch_size = 32
test_batch_size = 32

num_steps = 35
l2_decay = 1e-5
lr_decay = 0.1
relu_dropout = 0.2
embedding_dropout = 0.2
attention_dropout = 0.2
residual_dropout = 0.2
num_blocks = 3
# due to the residual connection, the embed_dim should be equal to hidden_size
decoder_hparams = {
    'share_embed_and_transform': True,
    'transform_with_bias': False,
    'beam_width': 1,
    'multiply_embedding_mode': 'sqrt_depth',
    'embedding_dropout': embedding_dropout,
    'attention_dropout': attention_dropout,
    'residual_dropout': residual_dropout,
    'sinusoid': True,
    'num_heads': 8,
    'num_units': hidden_size,
    'zero_pad': False,
    'bos_pad': False,
    'initializer': {
        'type': 'variance_scaling_initializer',
        'kwargs': {
            'scale': 1.0,
            'mode':'fan_avg',
            'distribution':'uniform',
        },
    },
    'poswise_feedforward': {
        'name':'fnn',
        'layers':[
            {
                'type':'Dense',
                'kwargs': {
                    'name':'conv1',
                    'units':hidden_size*4,
                    'activation':'relu',
                    'use_bias':True,
                },
            },
            {
                'type':'Dropout',
                'kwargs': {
                    'rate': relu_dropout,
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
    }
}
emb = {
    'name': 'lookup_table',
    "dim": embed_dim,
    'initializer' : {
        'type': 'random_normal_initializer',
        'kwargs': {
            'mean': 0.0,
            'stddev': embed_dim**-0.5,
        },
    }
}

opt = {
    'init_lr': 0.003,
    'Adam_beta1': 0.9,
    'Adam_beta2': 0.999,
    'epsilon': 1e-9,
}
