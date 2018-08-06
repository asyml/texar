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
    }
}
