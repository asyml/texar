"""
Utils to preprocess data for text style transfer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import Counter

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("path", "../../data/yelp", "data folder")
flags.DEFINE_string("train", "sentiment.train", "train file")
flags.DEFINE_string("val", "sentiment.dev", "val file")
flags.DEFINE_string("test", "sentiment.test", "test file")
flags.DEFINE_integer("min_word_count", 5, "min number of word count")

def load_data(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(line.split())
    return data

def makeup(_x, n):
    x = []
    for i in range(n):
        x.append(_x[i % len(_x)])
    return x

def write_sent(sents, path):
    with open(path, "w") as f:
        for sent in sents:
            f.write(" ".join(sent) + "\n")

def write_list(l, path):
    with open(path, "w") as f:
        for e in l:
            f.write(str(e) + "\n")

def build_vocab(data, min_count=5):
    id2word = []
    words = [word for sent in data for word in sent]
    cnt = Counter(words)
    for word in cnt.most_common():
        if word[1] >= min_count:
            id2word.append(word[0])
    write_list(id2word, os.path.join(FLAGS.path, "vocab"))

def sort_data(data0, data1, prefix):
    n = max(len(data0), len(data1))
    if len(data0) < n:
        data0 = makeup(data0, n)
    if len(data1) < n:
        data1 = makeup(data1, n)
    order0, order1 = range(n), range(n)
    z = sorted(zip(order0, data0), key=lambda i: len(i[1]))
    order0, data0 = zip(*z)
    z = sorted(zip(order1, data1), key=lambda i: len(i[1]))
    order1, data1 = zip(*z)
    write_sent(data0, prefix + ".sort.0")
    write_sent(data1, prefix + ".sort.1")
    write_list(order0, prefix + ".order.0")
    write_list(order1, prefix + ".order.1")

def main(unused_args):
    train0 = load_data(os.path.join(FLAGS.path, FLAGS.train + ".0"))
    train1 = load_data(os.path.join(FLAGS.path, FLAGS.train + ".1"))
    build_vocab(train0 + train1, FLAGS.min_word_count)
    sort_data(train0, train1, os.path.join(FLAGS.path, FLAGS.train))

    val0 = load_data(os.path.join(FLAGS.path, FLAGS.val + ".0"))
    val1 = load_data(os.path.join(FLAGS.path, FLAGS.val + ".1"))
    sort_data(val0, val1, os.path.join(FLAGS.path, FLAGS.val))

    test0 = load_data(os.path.join(FLAGS.path, FLAGS.test + ".0"))
    test1 = load_data(os.path.join(FLAGS.path, FLAGS.test + ".1"))
    sort_data(test0, test1, os.path.join(FLAGS.path, FLAGS.test))

if __name__ == "__main__":
    tf.app.run()
