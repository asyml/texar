# The full possible hyperparameters for the attentional seq2seq model.
# Most of the hyperparameters take the default values and are not necessary to
# specify explicitly. The config here results in the same model with the
# `config_model.py`.

num_units = 256
beam_width = 10

# --------------------- Embedder --------------------- #
embedder = {
    'dim': num_units,
    'initializer': {
        'type': 'random_uniform_initializer',
        'kwargs': {
            'minval': -0.1,
            'maxval': 0.1,
            'seed': None
        },
    },
    'regularizer': {
        'type': 'L1L2',
        'kwargs': {
            'l1': 0,
            'l2': 0
        }
    },
    'dropout_rate': 0,
    'dropout_strategy': 'element',
    'trainable': True,
    'name': 'word_embedder'
}

# --------------------- Encoder --------------------- #
encoder = {
    'rnn_cell_fw': {
        'type': 'LSTMCell',
        'kwargs': {
            'num_units': num_units,
            'forget_bias': 1.0,
            'activation': None,
            # Other arguments go here for tf.nn.rnn_cell.LSTMCell
            # ...
        },
        'num_layers': 1,
        'dropout': {
            'input_keep_prob': 1.0,
            'output_keep_prob': 1.0,
            'state_keep_prob': 1.0,
            'variational_recurrent': False,
            'input_size': [],
        },
        'residual': False,
        'highway': False,
    },
    'rnn_cell_bw': {
        # The same possible hyperparameters as with 'rnn_cell_fw'
        # ...
    },
    'rnn_cell_share_config': True,
    'output_layer_fw': {
        'num_layers': 0,
        'layer_size': 128,
        'activation': 'identity',
        'final_layer_activation': None,
        'other_dense_kwargs': None,
        'dropout_layer_ids': [],
        'dropout_rate': 0.5,
        'variational_dropout': False
    },
    'output_layer_bw': {
        # The same possible hyperparameters as with 'output_layer_fw'
        # ...
    },
    'output_layer_share_config': True,
    'name': 'bidirectional_rnn_encoder'
}

# --------------------- Decoder --------------------- #
decoder = {
    'rnn_cell': {
        'type': 'LSTMCell',
        'kwargs': {
            'num_units': num_units,
            'forget_bias': 1.0,
            'activation': None,
            # Other arguments go here for tf.nn.rnn_cell.LSTMCell
            # ...
        },
        'num_layers': 1,
        'dropout': {
            'input_keep_prob': 1.0,
            'output_keep_prob': 1.0,
            'state_keep_prob': 1.0,
            'variational_recurrent': False,
            'input_size': [],
        },
        'residual': False,
        'highway': False,
    },
    'attention': {
        'type': 'LuongAttention',
        'kwargs': {
            'num_units': num_units,
            'scale': False,
            'probability_fn': None,
            'score_mask_value': None,
            # Other arguments go here for tf.contrib.seq2seq.LuongAttention
            # ...
        },
        'attention_layer_size': num_units,
        'alignment_history': False,
        'output_attention': True,
    },
    'helper_train': {
        'type': 'TrainingHelper',
        'kwargs': {
            # Arguments go here for tf.contrib.seq2seq.TrainingHelper
        }
    },
    'helper_infer': {
        # The same possible hyperparameters as with 'helper_train'
        # ...
    },
    'max_decoding_length_train': None,
    'max_decoding_length_infer': None,
    'name': 'attention_rnn_decoder'
}
