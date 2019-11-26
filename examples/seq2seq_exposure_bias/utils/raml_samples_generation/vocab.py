import argparse
from collections import Counter
from itertools import chain

import torch

from util import read_corpus


class VocabEntry(object):
    def __init__(self):
        self.word2id = dict()
        self.unk_id = 3
        self.word2id['<pad>'] = 0
        self.word2id['<s>'] = 1
        self.word2id['</s>'] = 2
        self.word2id['<unk>'] = 3

        self.id2word = {v: k for k, v in self.word2id.iteritems()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    @staticmethod
    def from_corpus(corpus, size, remove_singleton=True):
        vocab_entry = VocabEntry()

        word_freq = Counter(chain(*corpus))
        non_singletons = [w for w in word_freq if word_freq[w] > 1]
        print('number of word types: %d, number of word types w/ frequency > 1: %d' % (len(word_freq),
                                                                                       len(non_singletons)))

        top_k_words = sorted(word_freq.keys(), reverse=True, key=word_freq.get)[:size]

        for word in top_k_words:
            if len(vocab_entry) < size:
                if not (word_freq[word] == 1 and remove_singleton):
                    vocab_entry.add(word)

        return vocab_entry


class Vocab(object):
    def __init__(self, src_sents, tgt_sents, src_vocab_size, tgt_vocab_size, remove_singleton=True):
        assert len(src_sents) == len(tgt_sents)

        print('initialize source vocabulary ..')
        self.src = VocabEntry.from_corpus(src_sents, src_vocab_size, remove_singleton=remove_singleton)

        print('initialize target vocabulary ..')
        self.tgt = VocabEntry.from_corpus(tgt_sents, tgt_vocab_size, remove_singleton=remove_singleton)

    def __repr__(self):
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_vocab_size', default=50000, type=int, help='source vocabulary size')
    parser.add_argument('--tgt_vocab_size', default=50000, type=int, help='target vocabulary size')
    parser.add_argument('--include_singleton', action='store_true', default=False, help='whether to include singleton'
                                                                                        'in the vocabulary (default=False)')

    parser.add_argument('--train_src', type=str, required=True, help='file of source sentences')
    parser.add_argument('--train_tgt', type=str, required=True, help='file of target sentences')

    parser.add_argument('--output', default='vocab.bin', type=str, help='output vocabulary file')

    args = parser.parse_args()

    print('read in source sentences: %s' % args.train_src)
    print('read in target sentences: %s' % args.train_tgt)

    src_sents = read_corpus(args.train_src, source='src')
    tgt_sents = read_corpus(args.train_tgt, source='tgt')

    vocab = Vocab(src_sents, tgt_sents, args.src_vocab_size, args.tgt_vocab_size, remove_singleton=not args.include_singleton)
    print('generated vocabulary, source %d words, target %d words' % (len(vocab.src), len(vocab.tgt)))

    torch.save(vocab, args.output)
    print('vocabulary saved to %s' % args.output)
