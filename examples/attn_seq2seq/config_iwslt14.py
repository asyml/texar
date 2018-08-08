
num_epochs = 15
display = 500

train = {
    'batch_size': 32,
    'allow_smaller_final_batch': False,
    'source_dataset': {
        "files": 'data/iwslt14/train.de',
        'vocab_file': 'data/iwslt14/vocab.de',
        'max_seq_length': 50
    },
    'target_dataset': {
        'files': 'data/iwslt14/train.en',
        'vocab_file': 'data/iwslt14/vocab.en',
        'max_seq_length': 50
    }
}
valid = {
    'batch_size': 32,
    'shuffle': False,
    'source_dataset': {
        "files": 'data/iwslt14/valid.de',
        'vocab_file': 'data/iwslt14/vocab.de'
    },
    'target_dataset': {
        'files': 'data/iwslt14/valid.en',
        'vocab_file': 'data/iwslt14/vocab.en'
    }
}
test = {
    'batch_size': 32,
    'shuffle': False,
    'source_dataset': {
        "files": 'data/iwslt14/test.de',
        'vocab_file': 'data/iwslt14/vocab.de'
    },
    'target_dataset': {
        'files': 'data/iwslt14/test.en',
        'vocab_file': 'data/iwslt14/vocab.en'
    }
}
