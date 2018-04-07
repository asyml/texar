#
"""
Example pipeline. This is a minimal example of basic RNN language model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name

import numpy as np
import tensorflow as tf
import texar as tx

train_data_hparams = {
    "num_epochs": 1,
    "seed": 123,
    "dataset": {
        "files": '/space/hzt/text/data/ptb/ptb.train.txt',
        "vocab_file": '/space/hzt/text/data/ptb/vocab.txt'
    }
}
test_data_hparams = {
    "num_epochs": 1,
    "dataset": {
        "files": '/space/hzt/text/data/ptb/ptb.test.txt', # TODO(zhiting): use new data
        "vocab_file": '/space/hzt/text/data/ptb/vocab.txt'
    }
}
opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {"learning_rate": 0.0001}
    },
    "gradient_clip": {
        "type": "clip_by_global_norm",
        "kwargs": {"clip_norm": 5.}
    },
}


def _main(_): #pylint: disable=too-many-locals
    # Data
    train_data = tx.data.MonoTextData(train_data_hparams)
    test_data = tx.data.MonoTextData(test_data_hparams)
    iterator = tx.data.TrainTestDataIterator(train=train_data,
                                             test=test_data)
    data_batch = iterator.get_next()

    # Model architecture
    embedder = tx.modules.WordEmbedder(
        vocab_size=train_data.vocab.size, hparams={"dim": 100})
    decoder = tx.modules.BasicRNNDecoder(vocab_size=train_data.vocab.size)
    outputs, _, seq_lengths = decoder(
        decoding_strategy="train_greedy",
        inputs=data_batch["text_ids"],
        sequence_length=data_batch["length"]-1,
        embedding=embedder)

    # Losses & train ops
    mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=data_batch['text_ids'][:, 1:],
        logits=outputs.logits,
        sequence_length=seq_lengths)
    train_op= tx.core.get_train_op(mle_loss, hparams=opt_hparams)

    # Prediction
    outputs_sample, _, _ = decoder(
        decoding_strategy="infer_sample",
        start_tokens=[test_data.vocab.bos_token_id]*5,
        end_token=test_data.vocab.eos_token_id,
        embedding=embedder)
    sample_text = test_data.vocab.map_ids_to_tokens(outputs_sample.sample_id)

    def _train_epochs(sess, epoch, display=1000):
        iterator.switch_to_train_data(sess)
        step = 0
        while True:
            try:
                fetches = {"train_op": train_op,
                           "mle_loss": mle_loss}
                feed = {tx.global_mode(): tf.estimator.ModeKeys.TRAIN}
                fetches_ = sess.run(fetches, feed_dict=feed)
                if step % display == 0:
                    print('epoch %d, step %d, loss %.4f' %
                          (epoch, step, fetches_["mle_loss"]))

                step += 1
            except tf.errors.OutOfRangeError:
                print('epoch %d, loss %.4f' % (epoch, fetches_["mle_loss"]))
                break

    def _test_epochs(sess, epoch):
        iterator.switch_to_test_data(sess)
        test_loss = []
        while True:
            try:
                fetches = {"mle_loss": mle_loss,
                           "sample": sample_text}
                feed = {tx.global_mode(): tf.estimator.ModeKeys.EVAL}
                fetches_ = sess.run(fetches, feed_dict=feed)
                test_loss.append(fetches_["mle_loss"])
            except tf.errors.OutOfRangeError:
                test_loss = np.mean(test_loss)
                print('[test] epoch %d, loss %.4f' % (epoch, test_loss))
                #print(fetches_["sample"])
                break

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        for epoch in range(100):
            _train_epochs(sess, epoch, display=100)
            _test_epochs(sess, epoch)

if __name__ == '__main__':
    tf.app.run(main=_main)

