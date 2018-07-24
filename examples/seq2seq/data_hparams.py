data_hparams = {
    'iwslt14': {
        'train': {
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
        },
        'valid': {
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
        },
        'test': {
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
    },
    'giga': {
        'train': {
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
        },
        'valid': {
            'num_epochs': 1,
            'batch_size': 32,
            'allow_smaller_final_batch': False,
            'shuffle': False,
            'source_dataset': {
                "files": ['data/giga/valid.article'],
                'vocab_file': 'data/giga/vocab.article',
            },
            'target_dataset': {
                'files': ['data/giga/valid.title'],
                'vocab_file': 'data/giga/vocab.title',
            }
        },
        'test': {
            'num_epochs': 1,
            'batch_size': 32,
            'allow_smaller_final_batch': False,
            'shuffle': False,
            'source_dataset': {
                "files": ['data/giga/test.article'],
                'vocab_file': 'data/giga/vocab.article',
            },
            'target_dataset': {
                'files': ['data/giga/test.title'],
                'vocab_file': 'data/giga/vocab.title',
            }
        }
    }
}
