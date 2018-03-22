#
"""
Example pipeline. This is a minimal example of basic RNN language model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name

data_hparams = {
    "num_epochs": 10,
    "seed": 123,
    "dataset": {
        "files": 'data/sent.txt',
        "vocab_file": 'data/vocab.txt'
    }
}

def main(_):
    pass

if __name__ == '__main__':
    pass
