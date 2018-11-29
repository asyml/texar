embed = {
    'dim': 768,
    'name': 'word_embeddings'
}
vocab_size = 30522

segment_embed = {
    'dim': 768,
    'name': 'token_type_embeddings'
}
type_vocab_size = 2

encoder = {
    'dim': 768,
    'embedding_dropout': 0.1,
    'multihead_attention': {
        'dropout_rate': 0.1,
        'name': 'self',
        'num_heads': 12,
        'num_units': 768,
        'output_dim': 768,
        'use_bias': True
    },
    'name': 'encoder',
    'num_blocks': 12,
    'position_embedder_hparams': {
        'dim': 768
    },
    'position_embedder_type': 'variables',
    'position_size': 512,
    'poswise_feedforward': {
        'layers': [
            {   'kwargs': {
                    'activation': 'gelu',
                    'name': 'intermediate',
                    'units': 3072,
                    'use_bias': True
                },
                'type': 'Dense'
            },
            {   'kwargs': {'activation': None,
                'name': 'output',
                'units': 768,
                'use_bias': True
                },
                'type': 'Dense'
            }
        ]
    },
    'residual_dropout': 0.1,
    'use_bert_config': True
}

output_size = 768 # The output dimension of BERT
