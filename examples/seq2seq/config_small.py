train_hparams = {
    'num_epochs': 1,
    'batch_size': 32,
    'allow_smaller_final_batch': False,
    'source_dataset': {
        "files": ['data/iwslt14/train.de'],
        'vocab_file': 'data/iwslt14/vocab.de',
        'max_seq_length': 50
    },
    'target_dataset': {
        'files': ['data/iwslt14/train.en'],
        'vocab_file': 'data/iwslt14/vocab.en',
        'max_seq_length': 50
    }
}

valid_hparams = {
    'num_epochs': 1,
    'batch_size': 32,
    'allow_smaller_final_batch': False,
    'shuffle': False,
    'source_dataset': {
        "files": ['data/iwslt14/valid.de'],
        'vocab_file': 'data/iwslt14/vocab.de'
    },
    'target_dataset': {
        'files': ['data/iwslt14/valid.en'],
        'vocab_file': 'data/iwslt14/vocab.en'
    }
}

test_hparams = {
    'num_epochs': 1,
    'batch_size': 32,
    'allow_smaller_final_batch': False,
    'shuffle': False,
    'source_dataset': {
        "files": ['data/iwslt14/test.de'],
        'vocab_file': 'data/iwslt14/vocab.de'
    },
    'target_dataset': {
        'files': ['data/iwslt14/test.en'],
        'vocab_file': 'data/iwslt14/vocab.en'
    }
}

num_epochs = 15

num_units = 256
beam_width = 10

encoder_hparams = {
    'rnn_cell_fw': {
        'kwargs': {
            'num_units': num_units
        }
    }
}

cell_hparams = {
    'kwargs': {
        'num_units': num_units
    },
}

embedder_hparams = {'dim': num_units}

decoder_hparams = {
    'attention': {
        'kwargs': {
            'num_units': num_units,
        },
        'attention_layer_size': num_units
    }
}
