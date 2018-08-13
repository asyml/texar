"""Attentional Seq2seq.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

#pylint: disable=invalid-name, too-many-arguments, too-many-locals

import importlib
import tensorflow as tf
import texar as tx

flags = tf.flags

flags.DEFINE_string("config_model", "config_model", "The model config.")
flags.DEFINE_string("config_data", "config_iwslt14", "The dataset config.")

FLAGS = flags.FLAGS

config_model = importlib.import_module(FLAGS.config_model)
config_data = importlib.import_module(FLAGS.config_data)


def build_model(data_batch, source_vocab_size, target_vocab_size,
                target_bos_token_id, target_eos_token_id):
    """Assembles the seq2seq model.
    """
    source_embedder = tx.modules.WordEmbedder(
        vocab_size=source_vocab_size, hparams=config_model.embedder)

    encoder = tx.modules.BidirectionalRNNEncoder(
        hparams=config_model.encoder)

    enc_outputs, _ = encoder(source_embedder(data_batch['source_text_ids']))

    embedder = tx.modules.WordEmbedder(
        vocab_size=target_vocab_size, hparams=config_model.embedder)

    decoder = tx.modules.AttentionRNNDecoder(
        memory=tx.modules.BidirectionalRNNEncoder.concat_outputs(enc_outputs),
        memory_sequence_length=data_batch['source_length'],
        vocab_size=target_vocab_size,
        hparams=config_model.decoder)

    training_outputs, _, _ = decoder(
        decoding_strategy='train_greedy',
        inputs=embedder(data_batch['target_text_ids'][:, :-1]),
        sequence_length=data_batch['target_length'] - 1)

    train_op = tx.core.get_train_op(
        tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=data_batch['target_text_ids'][:, 1:],
            logits=training_outputs.logits,
            sequence_length=data_batch['target_length'] - 1))

    start_tokens = \
        tf.ones_like(data_batch['target_length']) * target_bos_token_id
    beam_search_outputs, _, _ = \
        tx.modules.beam_search_decode(
            decoder_or_cell=decoder,
            embedding=embedder,
            start_tokens=start_tokens,
            end_token=target_eos_token_id,
            beam_width=config_model.beam_width,
            max_decoding_length=60)

    return train_op, beam_search_outputs


def main():
    """Entrypoint.
    """
    training_data = tx.data.PairedTextData(hparams=config_data.train)
    valid_data = tx.data.PairedTextData(hparams=config_data.valid)
    test_data = tx.data.PairedTextData(hparams=config_data.test)
    data_iterator = tx.data.TrainTestDataIterator(
        train=training_data, val=valid_data, test=test_data)

    source_vocab_size = training_data.source_vocab.size
    target_vocab_size = training_data.target_vocab.size
    target_bos_token_id = training_data.target_vocab.bos_token_id
    target_eos_token_id = training_data.target_vocab.eos_token_id

    data_batch = data_iterator.get_next()

    train_op, infer_outputs = build_model(
        data_batch, source_vocab_size, target_vocab_size,
        target_bos_token_id, target_eos_token_id)

    def _train_epoch(sess):
        data_iterator.switch_to_train_data(sess)

        step = 0
        while True:
            try:
                loss = sess.run(train_op)
                if step % config_data.display == 0:
                    print("step={}, loss={:.4f}".format(step, loss))
                step += 1
            except tf.errors.OutOfRangeError:
                break

    def _eval_epoch(sess, mode):
        if mode == 'valid':
            data_iterator.switch_to_val_data(sess)
        else:
            data_iterator.switch_to_test_data(sess)

        refs, hypos = [], []
        while True:
            try:
                fetches = [
                    data_batch['target_text'][:, 1:],
                    infer_outputs.predicted_ids[:, :, 0]
                ]
                feed_dict = {
                    tx.global_mode(): tf.estimator.ModeKeys.PREDICT
                }
                target_texts_ori, output_ids = \
                    sess.run(fetches, feed_dict=feed_dict)

                target_texts = tx.utils.strip_special_tokens(target_texts_ori)
                output_texts = tx.utils.map_ids_to_strs(
                    ids=output_ids, vocab=valid_data.target_vocab)

                for hypo, ref in zip(output_texts, target_texts):
                    hypos.append(hypo)
                    refs.append([ref])
            except tf.errors.OutOfRangeError:
                break

        return tx.evals.corpus_bleu(list_of_references=refs, hypotheses=hypos)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        best_valid_bleu = -1.
        for i in range(config_data.num_epochs):
            _train_epoch(sess)

            valid_bleu = _eval_epoch(sess, 'valid')
            best_valid_bleu = max(best_valid_bleu, valid_bleu)
            print('valid epoch={}, BLEU={:.4f}; best-ever={:.4f}'.format(
                i, valid_bleu, best_valid_bleu))

            test_bleu = _eval_epoch(sess, 'test')
            print('test epoch={}, BLEU={:.4f}'.format(i, test_bleu))

            print('=' * 50)


if __name__ == '__main__':
    main()

