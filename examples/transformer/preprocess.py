from __future__ import unicode_literals
import collections
import io
import re
import json
import os
from config import get_preprocess_args


split_pattern = re.compile(r'([.,!?"\':;)(])')

digit_pattern = re.compile(r'\d')

def split_sentence(s, tok=False):
    if tok:
        s = s.lower()
        s = s.replace('\u2019', "'")
        s = digit_pattern.sub('0', s)
    words = []
    for word in s.split():
        if tok:
            words.extend(split_pattern.split(word))
        else:
            words.append(word)
    words = [w for w in words if w]
    return words


def open_file(path):
    return io.open(path, encoding='utf-8', errors='ignore')

def read_file(path, tok=False):
    with open_file(path) as f:
        for line in f.readlines():
            words = split_sentence(line.strip(), tok)
            yield words


def count_words(path, max_vocab_size=40000, tok=False):
    counts = collections.Counter()
    for words in read_file(path, tok):
        for word in words:
            counts[word] += 1

    vocab = [word for (word, _) in counts.most_common(max_vocab_size)]
    return vocab


def make_dataset(path, w2id, tok=False):
    dataset = []
    for words in read_file(path, tok):
        dataset.append(words)
    return dataset

if __name__ == "__main__":
    args = get_preprocess_args()

    print(json.dumps(args.__dict__, indent=4))

    # Vocab Construction
    source_path = os.path.join(args.input_dir, args.source_train)
    target_path = os.path.join(args.input_dir, args.target_train)

    src_cntr = count_words(source_path, args.source_vocab, args.tok)
    trg_cntr = count_words(target_path, args.target_vocab, args.tok)
    all_words = sorted(list(set(src_cntr + trg_cntr)))

    vocab = all_words

    w2id = {word: index for index, word in enumerate(vocab)}

    # Train Dataset
    source_data = make_dataset(source_path, w2id, args.tok)
    target_data = make_dataset(target_path, w2id, args.tok)
    assert len(source_data) == len(target_data)
    train_data = [(s, t) for s, t in zip(source_data, target_data)
                  if 0 < len(s) < args.max_seq_length
                  and 0 < len(t) < args.max_seq_length]

    # Display corpus statistics
    print("Vocab: {}".format(len(vocab)))
    print('Original training data size: %d' % len(source_data))
    print('Filtered training data size: %d' % len(train_data))

    # Valid Dataset
    source_path = os.path.join(args.input_dir, args.source_valid)
    source_data = make_dataset(source_path, w2id, args.tok)
    target_path = os.path.join(args.input_dir, args.target_valid)
    target_data = make_dataset(target_path, w2id, args.tok)
    assert len(source_data) == len(target_data)
    valid_data = [(s, t) for s, t in zip(source_data, target_data)
                  if 0 < len(s) and 0 < len(t)]

    # Test Dataset
    source_path = os.path.join(args.input_dir, args.source_test)
    source_data = make_dataset(source_path, w2id, args.tok)
    target_path = os.path.realpath(os.path.join(args.input_dir, args.target_test))
    target_data = make_dataset(target_path, w2id, args.tok)
    assert len(source_data) == len(target_data)
    test_data = [(s, t) for s, t in zip(source_data, target_data)
                 if 0 < len(s) and 0 < len(t)]

    id2w = {i: w for w, i in w2id.items()}
    # Save the dataset to numpy files
    train_src_output = os.path.join(args.input_dir, args.save_data + 'train.' + args.src+ '.txt')
    train_tgt_output = os.path.join(args.input_dir, args.save_data + 'train.' + args.tgt + '.txt')
    dev_src_output = os.path.join(args.input_dir, args.save_data + 'dev.' + args.src+ '.txt')
    dev_tgt_output = os.path.join(args.input_dir, args.save_data + 'dev.' + args.tgt+ '.txt')
    test_src_output = os.path.join(args.input_dir, args.save_data + 'test.' + args.src+ '.txt')
    test_tgt_output = os.path.join(args.input_dir, args.save_data + 'test.' + args.tgt + '.txt')

    with open(train_src_output, 'w+') as fsrc, open(train_tgt_output, 'w+') as ftgt:
        for words in train_data:
            fsrc.write('{}\n'.format(' '.join(words[0])))
            ftgt.write('{}\n'.format(' '.join(words[1])))
    with open(dev_src_output, 'w+') as fsrc, open(dev_tgt_output, 'w+') as ftgt:
        for words in valid_data:
            fsrc.write('{}\n'.format(' '.join(words[0])))
            ftgt.write('{}\n'.format(' '.join(words[1])))
    with open(test_src_output, 'w+') as fsrc, open(test_tgt_output, 'w+') as ftgt:
        for words in test_data:
            fsrc.write('{}\n'.format(' '.join(words[0])))
            ftgt.write('{}\n'.format(' '.join(words[1])))
    with open(os.path.join(args.input_dir,
            args.save_data + args.pre_encoding + '.vocab.text'.format(args.src, args.tgt)), 'w+') as f:
        max_size = len(id2w)
        for idx in range(max_size):
            f.write('{}\n'.format(id2w[idx]))
