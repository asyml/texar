num_epochs = 4
display = 50

source_vocab_file = './data/toy_copy/train/vocab.sources.txt'
target_vocab_file = './data/toy_copy/train/vocab.targets.txt'

train = {
    'batch_size': 32,
    'source_dataset': {
        "files": './data/toy_copy/train/sources.txt',
        'vocab_file': source_vocab_file
    },
    'target_dataset': {
        'files': './data/toy_copy/train/targets.txt',
        'vocab_file': target_vocab_file
    }
}
val = {
    'batch_size': 32,
    'source_dataset': {
        "files": './data/toy_copy/dev/sources.txt',
        'vocab_file': source_vocab_file
    },
    'target_dataset': {
        "files": './data/toy_copy/dev/targets.txt',
        'vocab_file': target_vocab_file
    }
}
test = {
    'batch_size': 32,
    'source_dataset': {
        "files": './data/toy_copy/test/sources.txt',
        'vocab_file': source_vocab_file
    },
    'target_dataset': {
        "files": './data/toy_copy/test/targets.txt',
        'vocab_file': target_vocab_file
    }
}
