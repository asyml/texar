# pylint: disable=invalid-name, too-few-public-methods, missing-docstring
init_lr = 0.003
init_scale = 0.04
num_epochs = 1000
hidden_size = 256
batch_size = 64
num_steps = 35
l2_decay = 1e-5
lr_decay = 0.1

decoder_hparams = {
    'share_embed_and_transform': True,
    'transform_with_bias': False,
    'beam_width': 1,
    'multiply_embedding_mode': 'sqrt_depth',
    'embedding_dropout': 0.1,
    'attention_dropout': 0.1,
    'residual_dropout': 0.1,
    'sinusoid': True,
    'num_blocks': 6,
    'num_heads': 8,
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
    }
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
