"""Configurations of Transformer model
"""
import copy
import texar as tx

random_seed = 1234
beam_width = 5
alpha = 0.6
hidden_dim = 768

emb = {
    'name': 'word_embeddings',
    'dim': hidden_dim,
    'initializer': {
        'type': 'random_normal_initializer',
        'kwargs': {
            'mean': 0.0,
            'stddev': hidden_dim**-0.5,
        },
    }
}

token_embed = {
    'name': 'token_type_embeddings',
    'dim': hidden_dim,
    'initializer': {
        'type': 'truncated_normal_initializer',
        'kwargs': {
            'stddev': 0.02,
        }
    }
}

## TODO: segment ids

encoder = {
    'dim': hidden_dim,
    'embed_scale': False,
    'position_embedder_type': 'variables',
    'embed_norm': True,
    'embed_scale': False,
    'num_blocks': 12,
    'position_size': 512,
    'multihead_attention': {
        'use_bias': True,
        'num_units': hidden_dim,
        'num_heads': 12,
        'output_dim': hidden_dim,
    },
    'dim': hidden_dim,
    'use_bert_config': True,
    'poswise_feedforward': {
        "layers": [
            {
                'type': 'Dense',
                'kwargs': {
                    'name': 'intermediate',
                    'units': hidden_dim*4,
                    'activation': 'gelu',
                    'use_bias': True,
                }
            },
            {
                'type': 'Dense',
                'kwargs': {
                    'name': 'output',
                    'units': hidden_dim,
                    'activation': None,
                    'use_bias': True,
                }
            },
        ],
    },
}
