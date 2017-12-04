"""
Utils to preprocess data for text style transfer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cPickle as pkl
import tensorflow as tf

import vocabulary
from utils import log_print

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

def data_to_id(data, word2id):
  data_id = [[word2id[word] if word in word2id else word2id["_UNK"]
              for word in sent ] for sent in data]
  return data_id

def main(unused_args):
  train0 = load_data(os.path.join(FLAGS.path, FLAGS.train + ".0"))
  train1 = load_data(os.path.join(FLAGS.path, FLAGS.train + ".1"))
  vocab = vocabulary.build_vocab(train0 + train1, FLAGS.min_word_count)
  log_print("vocab size %d" %(vocab["size"]))
  with open(os.path.join(FLAGS.path, "vocab.pkl"), "w") as f:
    pkl.dump(vocab, f, pkl.HIGHEST_PROTOCOL)

  train0_id = data_to_id(train0, vocab["word2id"])
  train1_id = data_to_id(train1, vocab["word2id"])
  with open(os.path.join(FLAGS.path, "train.pkl"), "w") as f:
    pkl.dump((train0_id, train1_id), f, pkl.HIGHEST_PROTOCOL)
  log_print("train0 size %d" %(len(train0_id)))
  log_print("train1 size %d" %(len(train1_id)))

  dev0 = load_data(os.path.join(FLAGS.path, FLAGS.val + ".0"))
  dev1 = load_data(os.path.join(FLAGS.path, FLAGS.val + ".1"))
  dev0_id = data_to_id(dev0, vocab["word2id"])
  dev1_id = data_to_id(dev1, vocab["word2id"])
  with open(os.path.join(FLAGS.path, "val.pkl"), "w") as f:
    pkl.dump((dev0_id, dev1_id), f, pkl.HIGHEST_PROTOCOL)
  log_print("dev0 size %d" %(len(dev0_id)))
  log_print("dev1 size %d" %(len(dev1_id)))

  test0 = load_data(os.path.join(FLAGS.path, FLAGS.test + ".0"))
  test1 = load_data(os.path.join(FLAGS.path, FLAGS.test + ".1"))
  test0_id = data_to_id(test0, vocab["word2id"])
  test1_id = data_to_id(test1, vocab["word2id"])
  with open(os.path.join(FLAGS.path, "test.pkl"), "w") as f:
    pkl.dump((test0_id, test1_id), f, pkl.HIGHEST_PROTOCOL)
  log_print("test0 size %d" %(len(test0_id)))
  log_print("test1 size %d" %(len(test1_id)))

if __name__ == "__main__":
  tf.app.run()
