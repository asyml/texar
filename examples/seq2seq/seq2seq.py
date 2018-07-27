from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import texar as tx

import os
import argparse
from rouge import Rouge

from data_hparams import data_hparams
import model_hparams

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--num_epochs', type=int, default=10)

arg_parser.add_argument('--dataset', type=str, choices=['iwslt14', 'giga'])
arg_parser.add_argument('--metric', type=str, choices=['bleu', 'rouge'])

args = arg_parser.parse_args()

log_dir = args.dataset + '_training_log/'
os.system('mkdir ' + log_dir)
valid_test_log_file = open(log_dir + 'valid_test_log.txt', 'w')


def loss_fn(data_batch, output):
    return tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=data_batch['target_text_ids'][:, 1:],
        logits=output.logits,
        sequence_length=data_batch['target_length'] - 1)


def encode(data_batch, vocab_size):
    embedder = tx.modules.WordEmbedder(
        vocab_size=vocab_size, hparams=model_hparams.embedder_hparams)

    encoder = tx.modules.BidirectionalRNNEncoder(
        hparams=model_hparams.encoder_hparams)

    enc_outputs, enc_last = \
        encoder(inputs=embedder(data_batch['source_text_ids']))

    return enc_outputs, enc_last


def build_model(data_batch, batch_size, source_vocab_size, target_vocab_size,
                target_bos_token_id, target_eos_token_id):
    enc_outputs, enc_last = encode(data_batch, source_vocab_size)

    decoder_cell = tx.core.layers.get_rnn_cell(
        hparams=model_hparams.cell_hparams)

    embedder = tx.modules.WordEmbedder(
        vocab_size=target_vocab_size, hparams=model_hparams.embedder_hparams)

    decoder = tx.modules.AttentionRNNDecoder(
        cell=decoder_cell,
        memory=tx.modules.BidirectionalRNNEncoder.concat_outputs(enc_outputs),
        memory_sequence_length=data_batch['source_length'],
        vocab_size=target_vocab_size,
        hparams=model_hparams.decoder_hparams)

    training_outputs, training_final_state, sequence_lengths = decoder(
        decoding_strategy='train_greedy',
        inputs=embedder(data_batch['target_text_ids'][:, :-1]),
        sequence_length=data_batch['target_length'] - 1,
        initial_state=decoder.zero_state(
            batch_size=batch_size, dtype=tf.float32))

    train_op = tx.core.get_train_op(loss_fn(data_batch, training_outputs))

    beam_search_outputs, beam_search_final_state = \
        tx.modules.beam_search_decode(
            decoder_or_cell=decoder,
            embedding=embedder,
            start_tokens=[target_bos_token_id] * batch_size,
            end_token=target_eos_token_id,
            beam_width=model_hparams.beam_width,
            max_decoding_length=60)

    return train_op, beam_search_outputs


def main():
    training_data = tx.data.PairedTextData(
        hparams=data_hparams[args.dataset]['train'])
    valid_data = tx.data.PairedTextData(
        hparams=data_hparams[args.dataset]['valid'])
    test_data = tx.data.PairedTextData(
        hparams=data_hparams[args.dataset]['test'])
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

    def _id2texts(ids, eos_token_id, id2token_dict):
        result = []
        for i in range(len(ids)):
            result.append([])
            for j in range(len(ids[i])):
                if ids[i][j] == eos_token_id:
                    break
                else:
                    result[-1].append(id2token_dict[ids[i][j]].encode('utf-8'))
        return result

    def _train_epoch(sess, epoch):
        data_iterator.switch_to_train_data(sess)
        log_file = open(log_dir + 'training_log' + str(epoch) + '.txt', 'w')

        counter = 0
        while True:
            try:
                counter += 1
                print(counter,
                      sess.run(train_op, feed_dict={
                          tx.global_mode(): tf.estimator.ModeKeys.TRAIN}),
                      file=log_file)
                log_file.flush()
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
                output_texts = _id2texts(
                    output_ids, target_eos_token_id,
                    valid_data.target_vocab.id_to_token_map_py)

                for i in range(len(target_texts)):
                    if args.metric == 'bleu':
                        refs.append(
                            [target_texts[i][
                             :target_texts[i].index('<EOS>')]])
                        hypos.append(output_texts[i])
                    else:
                        refs.append(' '.join(
                            target_texts[i][:target_texts[i].index(
                                '<EOS>')]).decode('utf-8'))
                        hypos.append(
                            ' '.join(output_texts[i]).decode('utf-8'))
            except tf.errors.OutOfRangeError:
                break

        if args.metric == 'bleu':
            return tx.evals.corpus_bleu(
                list_of_references=refs, hypotheses=hypos)
        else:
            rouge = Rouge()
            return rouge.get_scores(hyps=hypos, refs=refs, avg=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        best_valid_score = -1.
        for i in range(args.num_epochs):
            _train_epoch(sess, i)

            if args.metric == 'bleu':
                valid_score = _eval_epoch(sess, 'valid')
                test_score = _eval_epoch(sess, 'test')

                best_valid_score = max(best_valid_score, valid_score)
                print('valid epoch', i, ':', valid_score,
                      'max ever: ', best_valid_score,
                      file=valid_test_log_file)
                print('test epoch', i, ':', test_score,
                      file=valid_test_log_file)
                print('=' * 100, file=valid_test_log_file)
                valid_test_log_file.flush()
            else:
                valid_score = _eval_epoch(sess, 'valid')
                test_score = _eval_epoch(sess, 'test')

                print('valid epoch', i, ':', file=valid_test_log_file)
                for key, value in valid_score.items():
                    print(key, value, file=valid_test_log_file)
                print('test epoch', i, ':', file=valid_test_log_file)
                for key, value in test_score.items():
                    print(key, value, file=valid_test_log_file)
                print('=' * 100, file=valid_test_log_file)
                valid_test_log_file.flush()


if __name__ == '__main__':
    main()
