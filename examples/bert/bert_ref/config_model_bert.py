"""Configurations of Transformer model
"""
random_seed = 1234
hidden_dim = 768
vocab_size =
emb = {
    'name': 'word_embeddings',
    'dim': hidden_dim,
}

token_embed = {
    'name': 'token_type_embeddings',
    'dim': hidden_dim,
}

encoder = {
    'name': 'encoder',
    'dim': hidden_dim,
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
        'dropout_rate': 0.1,
        'name': 'self'
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
