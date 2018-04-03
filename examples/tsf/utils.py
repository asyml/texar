"""Utils for tsf."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import random
import numpy as np

def log_print(line):
    """Add time to print function."""
    print(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time()))
          + "] " + line)


def data_to_id(data, word2id):
    data_id = [[word2id[word] if word in word2id else word2id["_UNK"]
                for word in sent ] for sent in data]
    return data_id

def write_sent(sents, path):
    with open(path, "w") as f:
        for sent in sents:
          f.write(" ".join(sent) + "\n")

def strip_eos(sents):
    return [sent[:sent.index("<EOS>")] if "<EOS>" in sent else sent
            for sent in sents]

def logits2word(logits, id2word):
    sents = np.argmax(logits, axis=2).tolist()
    sents = [[id2word[word] for word in sent] for sent in sents]
    return strip_eos(sents)

def write_sent(sents, path):
    with open(path, "w") as f:
      for sent in sents:
        f.write(" ".join(sent) + "\n")

class Stats():
    def __init__(self):
        self.reset()

    def reset(self):
        self._loss, self._g, self._ppl, self._d, self._d0, self._d1 \
            = [], [], [], [], [], []
        self._w_loss, self._w_g, self._w_ppl, self._w_d, self._w_d0, self._w_d1 \
            = 0, 0, 0, 0, 0, 0

    def append(self, loss, g, ppl, d, d0, d1,
               w_loss=1., w_g=1., w_ppl=1., w_d=1, w_d0=1., w_d1=1.):
        self._loss.append(loss*w_loss)
        self._g.append(g*w_g)
        self._ppl.append(ppl*w_ppl)
        self._d.append(d*w_d)
        self._d0.append(d0*w_d0)
        self._d1.append(d1*w_d1)
        self._w_loss += w_loss
        self._w_g += w_g
        self._w_ppl += w_ppl
        self._w_d += w_d
        self._w_d0 += w_d0
        self._w_d1 += w_d1

    @property
    def loss(self):
        return sum(self._loss) / self._w_loss

    @property
    def g(self):
        return sum(self._g) / self._w_g

    @property
    def ppl(self):
        return sum(self._ppl) / self._w_ppl

    @property
    def d(self):
        return sum(self._d) / self._w_d

    @property
    def d0(self):
        return sum(self._d0) / self._w_d0

    @property
    def d1(self):
        return sum(self._d1) / self._w_d1

    def __str__(self):
        return "loss %.2f, g %.2f, ppl %.2f d %.2f, adv %.2f %.2f" %(
            self.loss, self.g, self.ppl, self.d, self.d0, self.d1)
