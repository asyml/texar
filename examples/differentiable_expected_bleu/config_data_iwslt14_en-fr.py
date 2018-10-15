source_vocab_file = 'data/iwslt14_en-fr/vocab.en'
target_vocab_file = 'data/iwslt14_en-fr/vocab.fr'

train = {
    'batch_size': 80,
    'allow_smaller_final_batch': False,
    'source_dataset': {
        "files": 'data/iwslt14_en-fr/train.en',
        'vocab_file': source_vocab_file,
        'max_seq_length': 50
    },
    'target_dataset': {
        'files': 'data/iwslt14_en-fr/train.fr',
        'vocab_file': target_vocab_file,
        'max_seq_length': 50
    },
}

val = {
    'batch_size': 80,
    'shuffle': False,
    'source_dataset': {
        "files": 'data/iwslt14_en-fr/valid.en',
        'vocab_file': source_vocab_file,
    },
    'target_dataset': {
        'files': 'data/iwslt14_en-fr/valid.fr',
        'vocab_file': target_vocab_file,
    },
}

test = {
    'batch_size': 80,
    'shuffle': False,
    'source_dataset': {
        "files": 'data/iwslt14_en-fr/test.en',
        'vocab_file': source_vocab_file,
    },
    'target_dataset': {
        'files': 'data/iwslt14_en-fr/test.fr',
        'vocab_file': target_vocab_file,
    },
}
