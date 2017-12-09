"""Utils for classifier"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import random

def get_batches(x, y, word2id, batch_size, min_len=5, shuffle=False):
  pad = word2id["_PAD"]

  if shuffle:
    order = range(len(x))
    random.shuffle(order)
    x = [ x[i] for i in order]
    y = [ y[i] for i in order]

  batches = []
  s = 0
  while s < len(x):
    t = min(s + batch_size, len(x))

    _x = []
    _y = []
    max_len = max([len(sent) for sent in x[s:t]])
    max_len = max(max_len, min_len)
    for i in range(batch_size):
      if s + i < t:
        sent = x[s+i]
        _x.append(sent + [pad] * (max_len - len(sent)))
        _y.append(y[s+i])
      else:
        _x.append([pad] * max_len)
        _y.append(0)

    batches.append({
      "x": _x,
      "y": _y,
      "actual_size": t - s
    })

    s = t

  return batches
  


