num_units = 256
beam_width = 5
decoder_layers = 1
dropout = 0.2

embedder = {
    'dim': num_units
}
encoder = {
    'rnn_cell_fw': {
        'kwargs': {
            'num_units': num_units
        },
        'dropout': {
            'input_keep_prob': 1. - dropout
        }
    }
}
decoder = {
    'rnn_cell': {
        'kwargs': {
            'num_units': num_units
        },
        'dropout': {
            'input_keep_prob': 1. - dropout
        },
        'num_layers': decoder_layers
    },
    'attention': {
        'kwargs': {
            'num_units': num_units,
        },
        'attention_layer_size': num_units
    }
}
opt = {
    'optimizer': {
        'type':  'AdamOptimizer',
        'kwargs': {
            'learning_rate': 0.001,
        },
    },
}
