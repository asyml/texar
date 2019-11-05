num_epochs = 50  # the best epoch occurs within 10 epochs in most cases
observe_steps = 500

eval_metric = 'bleu'

batch_size = 64
source_vocab_file = './data/iwslt14/vocab.de'
target_vocab_file = './data/iwslt14/vocab.en'

train = {
    'batch_size': batch_size,
    'shuffle': True,
    'allow_smaller_final_batch': False,
    'source_dataset': {
        "files": 'data/iwslt14/train.de',
        'vocab_file': source_vocab_file,
        'max_seq_length': 50
    },
    'target_dataset': {
        'files': 'data/iwslt14/train.en',
        'vocab_file': target_vocab_file,
        'max_seq_length': 50
    }
}
val = {
    'batch_size': batch_size,
    'shuffle': False,
    'allow_smaller_final_batch': True,
    'source_dataset': {
        "files": 'data/iwslt14/valid.de',
        'vocab_file': source_vocab_file,
    },
    'target_dataset': {
        'files': 'data/iwslt14/valid.en',
        'vocab_file': target_vocab_file,
    }
}
test = {
    'batch_size': batch_size,
    'shuffle': False,
    'allow_smaller_final_batch': True,
    'source_dataset': {
        "files": 'data/iwslt14/test.de',
        'vocab_file': source_vocab_file,
    },
    'target_dataset': {
        'files': 'data/iwslt14/test.en',
        'vocab_file': target_vocab_file,
    }
}
