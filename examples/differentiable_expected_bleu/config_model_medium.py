# Attentional Seq2seq model.
# Hyperparameters not specified here will take the default values.

num_units = 256
embedding_dim = 256
dropout = 0.2

embedder = {
    'dim': embedding_dim
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

connector = None

decoder = {
    'rnn_cell': {
        'kwargs': {
            'num_units': num_units
        },
        'dropout': {
            'input_keep_prob': 1. - dropout
        }
    },
    'attention': {
        'kwargs': {
            'num_units': num_units,
        },
        'attention_layer_size': num_units
    }
}
