#
"""
Example pipeline. This is a minimal example of basic RNN language model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name

import texar as tx

train_data_hparams = {
    "num_epochs": 10,
    "seed": 123,
    "dataset": {
        "files": 'data/sent.txt',
        "vocab_file": 'data/vocab.txt'
    }
}
test_data_hparams = {
    "num_epochs": 1,
    "dataset": {
        "files": 'data/sent.txt', # TODO(zhiting): use new data
        "vocab_file": 'data/vocab.txt'
    }
}

def _main(_):
    train_data = tx.data.MonoTextData(train_data_hparams)
    test_data = tx.data.MonoTextData(test_data_hparams)
    iterator = tx.data.TrainTestDataIterator(train=train_data,
                                             test=test_data)
    data_batch = iterator.get_next()

    embedder = tx.modules.WordEmbedder(
        vocab_size=train_data.vocab.size, hparams={"dim": 100})

    decoder = tx.modules.BasicRNNDecoder(vocab_size=train_data.vocab.size)

    helper_train = tx.modules.get_helper(
        decoder.hparams.helper_train.type,
        inputs=embedder(data_batch["text_ids"]),
        sequence_length=data_batch["length"]-1)
    outputs_train, _, seq_lengths_train = decoder(helper=helper_train)

    #mle_loss = mle_losses.average_sequence_sparse_softmax_cross_entropy(
    #    labels=data_batch['text_ids'][:, 1:],
    #    logits=outputs.logits,
    #    sequence_length=sequence_lengths)


if __name__ == '__main__':
    pass
