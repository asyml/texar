"""Some utils.."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import time
import random
import numpy as np

def log_print(line):
  """Add time to print function."""
  print(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time()))
        + "] " + line)

def makeup(_x, n):
  x = []
  for i in range(n):
    x.append(_x[i % len(_x)])
  return x

def get_batch(x, y, word2id, batch_size, min_len=5):
  pad = word2id["_PAD"]
  go = word2id["_GO"]
  eos = word2id["_EOS"]
  unk = word2id["_UNK"]

  max_len = max([len(sent) for sent in x])
  max_len = max(max_len, min_len)

  rev_x, go_x, x_eos, weights = [], [], [], []
  for sent in x:
    l = len(sent)
    padding = [pad] * (max_len - l)
    rev_x.append(padding + sent[::-1])
    go_x.append([go] + sent + padding)
    x_eos.append(sent + [eos] + padding)
    weights.append([1.] * (l+1) + [0.] * (max_len - l))

  return {
    "enc_inputs": rev_x,
    "dec_inputs": go_x,
    "targets": x_eos,
    "weights": weights,
    "labels": y,
    "size": len(x),
    "len": max_len + 1
  }

def get_batches(x0, x1, word2id, batch_size, shuffle=True):
  # half as batch size
  batch_size = batch_size // 2
  if len(x0) < len(x1):
    x0 = makeup(x0, len(x1))
  if len(x1) < len(x0):
    x1 = makeup(x1, len(x0))
  n = len(x0)

  if shuffle:
    random.shuffle(x0)
    random.shuffle(x1)

  # order0 = range(n)
  # z = sorted(zip(order0, x0), key=lambda i:len(i[1]))
  # order0, x0 = zip(*z)

  # order1 = range(n)
  # z = sorted(zip(order1, x1), key=lambda i:len(i[1]))
  # order1, x1 = zip(*z)

  batches = []
  s = 0
  while s < n:
    t = min(s + batch_size, n)
    if t < s + batch_size:
      break
    batches.append(get_batch(x0[s:t] + x1[s:t],
                             [0] * (t-s) + [1]*(t-s),
                             word2id,
                             batch_size))
    s = t

  if shuffle:
    random.shuffle(batches)
 
  return batches

def strip_eos(sents):
  return [sent[:sent.index("_EOS")] if "_EOS" in sent else sent
          for sent in sents]

def logits2word(logits, id2word):
  sents = np.argmax(logits, axis=2).tolist()
  sents = [[id2word[word] for word in sent] for sent in sents]
  return strip_eos(sents)

def write_sent(sents, path):
  with open(path, "w") as f:
    for sent in sents:
      f.write(" ".join(sent) + "\n")
