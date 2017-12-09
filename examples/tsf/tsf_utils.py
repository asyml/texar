"""Utils for TSF."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import random
import numpy as np


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

def get_batches(x0, x1, word2id, batch_size, sort=False):
  # half as batch size
  batch_size = batch_size // 2
  n = max(len(x0), len(x1))
  n = (n // batch_size + 1) * batch_size
  if len(x0) < n:
    x0 = makeup(x0, n)
  if len(x1) < n:
    x1 = makeup(x1, n)

  if sort:
    order0 = range(n)
    z = sorted(zip(order0, x0), key=lambda i:len(i[1]))
    order0, x0 = zip(*z)

    order1 = range(n)
    z = sorted(zip(order1, x1), key=lambda i:len(i[1]))
    order1, x1 = zip(*z)
  else:
    order0 = range(n)
    order1 = range(n)
    random.shuffle(order0)
    random.shuffle(order1)
    x0 = [x0[i] for i in order0]
    x1 = [x1[i] for i in order1]

  batches = []
  s = 0
  while s < n:
    t = s + batch_size
    batches.append(get_batch(x0[s:t] + x1[s:t],
                             [0] * (t-s) + [1]*(t-s),
                             word2id,
                             batch_size))
    s = t
 
  return batches, order0, order1

