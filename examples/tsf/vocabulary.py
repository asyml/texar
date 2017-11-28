"""
Vocabulary for text style transfer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

class Vocabulary:
  pass

def build_vocab(data, min_count=5):
  word2id = {"_PAD": 0, "_GO": 1, "_EOS": 2, '_UNK': 3}
  id2word = ["_PAD", "_GO", "_EOS", "_UNK"]

  words = [word for sent in data for word in sent]
  cnt = Counter(words)
  for word in cnt.most_common():
    if word[1] >= min_count:
      word2id[word[0]] = len(word2id)
      id2word.append(word[0])

  return {
    "size": len(word2id),
    "word2id": word2id,
    "id2word": id2word
  }


