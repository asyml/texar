import tensorflow as tf
import numpy as np
import codecs
import regex

data_hparams = {
    'vocab_min_cnt':20,
    'max_seq_length':10,
    'source_train':'data/translation/de-en/train_de_sentences.txt',
    'target_train':'data/translation/de-en/train_en_sentences.txt',
    'source_vocab':'data/translation/de-en/filter_de.vocab.txt',
    'target_vocab':'data/translation/de-en/filter_en.vocab.txt',

    'source_test':'data/translation/de-en/IWSLT16.TED.tst2014.de-en.de.xml',
    'target_test':'data/translation/de-en/IWSLT16.TED.tst2014.de-en.en.xml',
    'vocab_file_path': './data/translation/de-en/',
    'batch_size':32,
    }
def load_de_vocab():
    vocab = [line.split()[0] for line in codecs.open(data_hparams['vocab_file_path']+'de.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=20]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open(data_hparams['vocab_file_path']+'en.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=20]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(source_sents, target_sents):
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()

    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [de2idx.get(word, 1) for word in (source_sent + u" </S>").split()]
        y = [en2idx.get(word, 1) for word in (target_sent + u" </S>").split()]
        if max(len(x), len(y)) <=data_hparams['max_seq_length']:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)

    X = np.zeros([len(x_list), data_hparams['max_seq_length']], np.int32)
    Y = np.zeros([len(y_list), data_hparams['max_seq_length']], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, data_hparams['max_seq_length']-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, data_hparams['max_seq_length']-len(y)], 'constant', constant_values=(0, 0))

    return X, Y, Sources, Targets

def load_train_data():
    de_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(data_hparams['source_train'], 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    en_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(data_hparams['target_train'], 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]

    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Y

def load_test_data():
    def _refine(line):
        line = regex.sub("<[^>]+>", "", line)
        line = regex.sub("[^\s\p{Latin}']", "", line)
        return line.strip()

    de_sents = [_refine(line) for line in codecs.open(data_hparams['source_test'], 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]
    en_sents = [_refine(line) for line in codecs.open(data_hparams['target_test'], 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]

    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Sources, Targets

def get_batch_data():
    X, Y = load_train_data()

    num_batch = len(X) // data_hparams['batch_size']

    X = tf.convert_to_tensor(X, tf.int64)
    Y = tf.convert_to_tensor(Y, tf.int64)

    input_queues = tf.train.slice_input_producer([X, Y])

    x, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=data_hparams['batch_size'],
                                capacity=data_hparams['batch_size']*64,
                                min_after_dequeue=data_hparams['batch_size']*32,
                                allow_smaller_final_batch=False,
                                seed=123)

    return x, y, num_batch # (N, T), (N, T), ()
