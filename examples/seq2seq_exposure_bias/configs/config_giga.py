num_epochs = 30
observe_steps = 500

eval_metric = 'rouge'

batch_size = 64
source_vocab_file = './data/giga/vocab.article'
target_vocab_file = './data/giga/vocab.title'

train = {
    'batch_size': batch_size,
    'allow_smaller_final_batch': False,
    'source_dataset': {
        "files": 'data/giga/train.article',
        'vocab_file': source_vocab_file
    },
    'target_dataset': {
        'files': 'data/giga/train.title',
        'vocab_file': target_vocab_file
    }
}
val = {
    'batch_size': batch_size,
    'shuffle': False,
    'allow_smaller_final_batch': True,
    'source_dataset': {
        "files": 'data/giga/valid.article',
        'vocab_file': source_vocab_file,
    },
    'target_dataset': {
        'files': 'data/giga/valid.title',
        'vocab_file': target_vocab_file,
    }
}
test = {
    'batch_size': batch_size,
    'shuffle': False,
    'allow_smaller_final_batch': True,
    'source_dataset': {
        "files": 'data/giga/test.article',
        'vocab_file': source_vocab_file,
    },
    'target_dataset': {
        'files': 'data/giga/test.title',
        'vocab_file': target_vocab_file,
    }
}
