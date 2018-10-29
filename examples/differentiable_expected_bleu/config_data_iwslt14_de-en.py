source_vocab_file = 'data/iwslt14_de-en/vocab.de'
target_vocab_file = 'data/iwslt14_de-en/vocab.en'

batch_size = 80

train = {
    'batch_size': batch_size,
    'allow_smaller_final_batch': False,
    'source_dataset': {
        "files": 'data/iwslt14_de-en/train.de',
        'vocab_file': source_vocab_file,
        'max_seq_length': 50
    },
    'target_dataset': {
        'files': 'data/iwslt14_de-en/train.en',
        'vocab_file': target_vocab_file,
        'max_seq_length': 50
    },
}

val = {
    'batch_size': batch_size,
    'shuffle': False,
    'source_dataset': {
        "files": 'data/iwslt14_de-en/valid.de',
        'vocab_file': source_vocab_file,
    },
    'target_dataset': {
        'files': 'data/iwslt14_de-en/valid.en',
        'vocab_file': target_vocab_file,
    },
}

test = {
    'batch_size': batch_size,
    'shuffle': False,
    'source_dataset': {
        "files": 'data/iwslt14_de-en/test.de',
        'vocab_file': source_vocab_file,
    },
    'target_dataset': {
        'files': 'data/iwslt14_de-en/test.en',
        'vocab_file': target_vocab_file,
    },
}
