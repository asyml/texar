# Attentional Seq2seq model.
# Hyperparameters not specified here will take the default values.

num_units = 256
beam_width = 10

embedder = {
    'dim': num_units
}
encoder = {
    'rnn_cell_fw': {
        'kwargs': {
            'num_units': num_units
        }
    }
}
decoder = {
    'rnn_cell': {
        'kwargs': {
            'num_units': num_units
        },
    },
    'attention': {
        'kwargs': {
            'num_units': num_units,
        },
        'attention_layer_size': num_units
    }
}
agent = {
    'discount_factor': 0.,
    'entropy_weight': .5
}
