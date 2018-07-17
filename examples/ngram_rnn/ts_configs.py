training_data_hparams = {
    'num_epochs': 1,
    'batch_size': 32,
    'allow_smaller_final_batch': False,
    'source_dataset': {
        "files": ['data/giga/train.article'],
        'vocab_file': 'data/giga/vocab.article',
    },
    'target_dataset': {
        'files': ['data/giga/train.title'],
        'vocab_file': 'data/giga/vocab.title',
    }
}

valid_data_hparams = {
    'num_epochs': 1,
    'batch_size': 32,
    'allow_smaller_final_batch': False,
    'source_dataset': {
        "files": ['data/giga/valid.article'],
        'vocab_file': 'data/giga/vocab.article',
    },
    'target_dataset': {
        'files': ['data/giga/valid.title'],
        'vocab_file': 'data/giga/vocab.title',
    }
}

test_data_hparams = {
    'num_epochs': 1,
    'batch_size': 32,
    'allow_smaller_final_batch': False,
    'source_dataset': {
        "files": ['data/giga/test.article'],
        'vocab_file': 'data/giga/vocab.article',
    },
    'target_dataset': {
        'files': ['data/giga/test.title'],
        'vocab_file': 'data/giga/vocab.title',
    }
}
