num_units = 512
dropout = 0.2
beam_width = 10

encoder_hparams = {
    'rnn_cell_fw': {
        'kwargs': {
            'num_units': num_units
        },
        'dropout': {
            'input_keep_prob': 1. - dropout
        }
    }
}

cell_hparams = {
    'kwargs': {
        'num_units': num_units
    },
    'num_layers': 2,
    'dropout': {
        'input_keep_prob': 1. - dropout
    }
}

embedder_hparams = {'dim': num_units}

decoder_hparams = {
    'attention': {
        'kwargs': {
            'num_units': num_units,
            'scale': True
        },
        'attention_layer_size': num_units
    }
}