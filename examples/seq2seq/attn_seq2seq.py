from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import texar as tx

import sys
import importlib

flags = tf.flags

flags.DEFINE_string("config", "config_small", "The config to use.")

FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)


def build_model(data_batch, batch_size, source_vocab_size, target_vocab_size,
                target_bos_token_id, target_eos_token_id):
    source_embedder = tx.modules.WordEmbedder(
        vocab_size=source_vocab_size, hparams=config.embedder_hparams)

    encoder = tx.modules.BidirectionalRNNEncoder(
        hparams=config.encoder_hparams)

    enc_outputs, enc_last = \
        encoder(inputs=source_embedder(data_batch['source_text_ids']))

    decoder_cell = tx.core.layers.get_rnn_cell(
        hparams=config.cell_hparams)

    embedder = tx.modules.WordEmbedder(
        vocab_size=target_vocab_size, hparams=config.embedder_hparams)

    decoder = tx.modules.AttentionRNNDecoder(
        cell=decoder_cell,
        memory=tx.modules.BidirectionalRNNEncoder.concat_outputs(enc_outputs),
        memory_sequence_length=data_batch['source_length'],
        vocab_size=target_vocab_size,
        hparams=config.decoder_hparams)

    training_outputs, training_final_state, sequence_lengths = decoder(
        decoding_strategy='train_greedy',
        inputs=embedder(data_batch['target_text_ids'][:, :-1]),
        sequence_length=data_batch['target_length'] - 1,
        initial_state=decoder.zero_state(
            batch_size=batch_size, dtype=tf.float32))

    train_op = tx.core.get_train_op(
        tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=data_batch['target_text_ids'][:, 1:],
            logits=training_outputs.logits,
            sequence_length=data_batch['target_length'] - 1))

    beam_search_outputs, beam_search_states, beam_search_lengths = \
        tx.modules.beam_search_decode(
            decoder_or_cell=decoder,
            embedding=embedder,
            start_tokens=[target_bos_token_id] * batch_size,
            end_token=target_eos_token_id,
            beam_width=config.beam_width,
            max_decoding_length=60)

    return train_op, beam_search_outputs


def main():
    training_data = tx.data.PairedTextData(hparams=config.train_hparams)
    valid_data = tx.data.PairedTextData(hparams=config.valid_hparams)
    test_data = tx.data.PairedTextData(hparams=config.test_hparams)
    data_iterator = tx.data.TrainTestDataIterator(
        train=training_data, val=valid_data, test=test_data)

    batch_size = training_data.batch_size
    source_vocab_size = training_data.source_vocab.size
    target_vocab_size = training_data.target_vocab.size
    target_bos_token_id = training_data.target_vocab.bos_token_id
    target_eos_token_id = training_data.target_vocab.eos_token_id

    data_batch = data_iterator.get_next()

    train_op, infer_outputs = build_model(
        data_batch, batch_size, source_vocab_size, target_vocab_size,
        target_bos_token_id, target_eos_token_id)

    def _train_epoch(sess):
        data_iterator.switch_to_train_data(sess)

        for counter in xrange(0, sys.maxint):
            try:
                loss = sess.run(train_op, feed_dict={
                    tx.global_mode(): tf.estimator.ModeKeys.TRAIN})
                if counter % 1000 == 0:
                    print(counter, loss)
            except tf.errors.OutOfRangeError:
                break

    def _eval_epoch(sess, mode):
        if mode == 'valid':
            data_iterator.switch_to_val_data(sess)
        else:
            data_iterator.switch_to_test_data(sess)

        refs = []
        hypos = []
        while True:
            try:
                target_texts, output_ids = sess.run(
                    [data_batch['target_text'][:, 1:],
                     infer_outputs.predicted_ids[:, :, 0]], feed_dict={
                        tx.global_mode(): tf.estimator.ModeKeys.PREDICT})

                target_texts = target_texts.tolist()
                output_texts = tx.utils.map_ids_to_strs(
                    ids=output_ids, vocab=valid_data.target_vocab, join=False)

                for j in range(len(target_texts)):
                    refs.append(
                        [target_texts[j][:target_texts[j].index('<EOS>')]])
                    hypos.append(output_texts[j])
            except tf.errors.OutOfRangeError:
                break

        return tx.evals.corpus_bleu(list_of_references=refs, hypotheses=hypos)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        best_valid_score = -1.
        for i in range(config.num_epochs):
            _train_epoch(sess)

            valid_score = _eval_epoch(sess, 'valid')
            test_score = _eval_epoch(sess, 'test')

            best_valid_score = max(best_valid_score, valid_score)
            print('valid epoch', i, ':', valid_score,
                  'max ever: ', best_valid_score)
            print('test epoch', i, ':', test_score)
            print('=' * 50)


if __name__ == '__main__':
    main()

