from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.contrib import seq2seq

import texar as tx
from ngram_decoder import NGramRNNDecoder

import sys
import os
import argparse
from nltk.translate import bleu_score
from rouge import Rouge

import mt_configs, ts_configs

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--k', type=int, default=1)
arg_parser.add_argument('--c', type=float, default=0.)
arg_parser.add_argument('--d', type=float, default=0.)
arg_parser.add_argument('--e', type=float, default=0.)
arg_parser.add_argument('--num_units', type=int, default=512)
arg_parser.add_argument('--num_epochs', type=int, default=10)
arg_parser.add_argument('--dropout', type=float, default=0.2)
arg_parser.add_argument('--beam_width', type=int, default=10)
arg_parser.add_argument('--task', type=str, choices=['mt', 'ts'])

args = arg_parser.parse_args()

log_dir = args.task + '_training_log' + '_k' + str(args.k) + \
          '_c' + str(args.c) + '_d' + str(args.d) + '_e' + str(args.e) + '/'
os.system('mkdir ' + log_dir)
valid_test_log_file = open(log_dir + 'valid_test_log.txt', 'w')


def encode(data_batch, vocab_size):
    embedder = tx.modules.WordEmbedder(
        vocab_size=vocab_size, hparams={'dim': args.num_units})

    encoder_hparams = tx.modules.BidirectionalRNNEncoder.default_hparams()
    encoder_hparams['rnn_cell_fw']['kwargs']['num_units'] = args.num_units
    encoder_hparams['rnn_cell_fw']['dropout']['input_keep_prob'] = \
        1. - args.dropout
    encoder = tx.modules.BidirectionalRNNEncoder(hparams=encoder_hparams)

    enc_outputs, enc_last = \
        encoder(inputs=embedder(data_batch['source_text_ids']))

    return enc_outputs, enc_last


def loss_fn(data_batch, output):
    mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=data_batch['target_text_ids'][:, 1:],
        logits=output.logits,
        sequence_length=data_batch['target_length'] - 1)

    loss_f1 = 0
    if args.k >= 2:
        loss_f1 = \
            tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=data_batch['target_text_ids'][:, 2:],
                logits=output.logits_f1[:, :-1],
                sequence_length=data_batch['target_length'] - 2) + \
            tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=output.sample_ids0[:, :-1],
                logits=output.logits[:, :-1],
                sequence_length=data_batch['target_length'] - 2)

    loss_f2 = 0
    if args.k >= 3:
        loss_f2 = \
            tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=data_batch['target_text_ids'][:, 3:],
                logits=output.logits_f2[:, :-2],
                sequence_length=data_batch['target_length'] - 3) + \
            tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=output.sample_ids1[:, :-2],
                logits=output.logits_f1[:, :-2],
                sequence_length=data_batch['target_length'] - 3) + \
            tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=output.sample_ids0[:, :-2],
                logits=output.logits[:, :-2],
                sequence_length=data_batch['target_length'] - 3)

    loss_f3 = 0
    if args.k >= 4:
        loss_f3 = \
            tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=data_batch['target_text_ids'][:, 4:],
                logits=output.logits_f3[:, :-3],
                sequence_length=data_batch['target_length'] - 4) + \
            tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=output.sample_ids2[:, :-3],
                logits=output.logits_f2[:, :-3],
                sequence_length=data_batch['target_length'] - 4) + \
            tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=output.sample_ids1[:, :-3],
                logits=output.logits_f1[:, :-3],
                sequence_length=data_batch['target_length'] - 4) + \
            tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=output.sample_ids0[:, :-3],
                logits=output.logits[:, :-3],
                sequence_length=data_batch['target_length'] - 4)

    return mle_loss + args.c * loss_f1 + args.d * loss_f2 + args.e * loss_f3


def get_train_op(data_batch, decoder_cell, embedder,
                 enc_outputs, enc_last, batch_size, target_vocab_size):
    decoder_hparams = NGramRNNDecoder.default_hparams()
    decoder_hparams['next_k'] = args.k
    decoder = NGramRNNDecoder(
        cell=decoder_cell,
        vocab_size=target_vocab_size,
        embedding=embedder,
        hparams=decoder_hparams)

    helper = tx.modules.get_helper(
        helper_type=decoder.hparams.helper_train.type,
        inputs=embedder(data_batch['target_text_ids'][:, :-1]),
        sequence_length=data_batch['target_length'] - 1)

    output, final_state, sequence_lengths = decoder(
        helper=helper, initial_state=decoder.zero_state(
            batch_size=batch_size, dtype=tf.float32))

    return decoder, tx.core.get_train_op(
        loss_fn(data_batch, output),
        hparams={'optimizer': {'type': 'AdamOptimizer'}})


def get_infer_outputs(data_batch, decoder_cell, embedder,
                      enc_outputs, enc_last, batch_size, target_vocab_size,
                      target_bos_token_id, target_eos_token_id,
                      training_decoder):
    decoder = seq2seq.BeamSearchDecoder(
        cell=decoder_cell,
        embedding=embedder,
        start_tokens=[target_bos_token_id] * batch_size,
        end_token=target_eos_token_id,
        beam_width=args.beam_width,
        initial_state=decoder_cell.zero_state(
            batch_size=batch_size * args.beam_width, dtype=tf.float32),
        output_layer=training_decoder.output_layer)

    outputs, _, _ = seq2seq.dynamic_decode(
        decoder=decoder, maximum_iterations=60)
    return outputs


def build_model(data_batch, batch_size, source_vocab_size, target_vocab_size,
                target_bos_token_id, target_eos_token_id):
    enc_outputs, enc_last = encode(data_batch, source_vocab_size)

    decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells=[
        tf.nn.rnn_cell.DropoutWrapper(
            cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=args.num_units),
            input_keep_prob=tx.utils.switch_dropout(1. - args.dropout)),
        tf.nn.rnn_cell.DropoutWrapper(
            cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=args.num_units),
            input_keep_prob=tx.utils.switch_dropout(1. - args.dropout))])

    attention_mechanism = seq2seq.LuongAttention(
        num_units=args.num_units,
        memory=tf.cond(
            tf.equal(tx.global_mode(), tf.estimator.ModeKeys.TRAIN),
            lambda: tx.modules.BidirectionalRNNEncoder.concat_outputs(
                enc_outputs),
            lambda: seq2seq.tile_batch(
                tx.modules.BidirectionalRNNEncoder.concat_outputs(enc_outputs),
                args.beam_width)),
        memory_sequence_length=tf.cond(
            tf.equal(tx.global_mode(), tf.estimator.ModeKeys.TRAIN),
            lambda: data_batch['source_length'],
            lambda: seq2seq.tile_batch(
                data_batch['source_length'], args.beam_width)),
        scale=True)

    decoder_cell = seq2seq.AttentionWrapper(
        cell=decoder_cell,
        attention_mechanism=attention_mechanism,
        attention_layer_size=args.num_units)

    embedder = tx.modules.WordEmbedder(
        vocab_size=target_vocab_size, hparams={'dim': args.num_units})

    training_decoder, train_op = get_train_op(
        data_batch, decoder_cell, embedder, enc_outputs, enc_last,
        batch_size, target_vocab_size)

    infer_outputs = get_infer_outputs(
        data_batch, decoder_cell, embedder, enc_outputs, enc_last,
        batch_size, target_vocab_size,
        target_bos_token_id, target_eos_token_id, training_decoder)

    return train_op, infer_outputs


def main():
    if args.task == 'mt':
        configs = mt_configs
    else:
        configs = ts_configs

    training_data = \
        tx.data.PairedTextData(hparams=configs.training_data_hparams)
    valid_data = tx.data.PairedTextData(hparams=configs.valid_data_hparams)
    test_data = tx.data.PairedTextData(hparams=configs.test_data_hparams)
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

    def train_epoch(sess, epoch):
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

    def id2texts(ids, eos_token_id, id2token_dict):
        result = []
        for i in range(len(ids)):
            result.append([])
            for j in range(len(ids[i])):
                if ids[i][j] == eos_token_id:
                    break
                else:
                    result[-1].append(id2token_dict[ids[i][j]].encode('utf-8'))
        return result

    def valid_epoch(sess):
        data_iterator.switch_to_val_data(sess)

        refs = []
        hypos = []
        while True:
            try:
                target_texts, output_ids = sess.run(
                    [data_batch['target_text'][:, 1:],
                     infer_outputs.predicted_ids[:, :, 0]], feed_dict={
                        tx.global_mode(): tf.estimator.ModeKeys.PREDICT})

                target_texts = target_texts.tolist()
                output_texts = id2texts(
                    output_ids, target_eos_token_id,
                    valid_data.target_vocab.id_to_token_map_py)

                for i in range(len(target_texts)):
                    if args.task == 'mt':
                        refs.append(
                            [target_texts[i][:target_texts[i].index('<EOS>')]])
                        hypos.append(output_texts[i])
                    else:
                        refs.append(' '.join(
                            target_texts[i][:target_texts[i].index(
                                '<EOS>')]).decode('utf-8'))
                        hypos.append(' '.join(output_texts[i]).decode('utf-8'))

                for i in range(batch_size):
                    print(refs[-i])
                    print(hypos[-i])

            except tf.errors.OutOfRangeError:
                break

        if args.task == 'mt':
            return 100. * bleu_score.corpus_bleu(
                list_of_references=refs, hypotheses=hypos)
        else:
            rouge = Rouge()
            return rouge.get_scores(hyps=hypos, refs=refs, avg=True)

    def test_epoch(sess):
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
                output_texts = id2texts(
                    output_ids, target_eos_token_id,
                    valid_data.target_vocab.id_to_token_map_py)

                for i in range(len(target_texts)):
                    if args.task == 'mt':
                        refs.append(
                            [target_texts[i][:target_texts[i].index('<EOS>')]])
                        hypos.append(output_texts[i])
                    else:
                        refs.append(' '.join(
                            target_texts[i][:target_texts[i].index(
                                '<EOS>')]).decode('utf-8'))
                        hypos.append(' '.join(output_texts[i]).decode('utf-8'))
            except tf.errors.OutOfRangeError:
                break

        if args.task == 'mt':
            return 100. * bleu_score.corpus_bleu(
                list_of_references=refs, hypotheses=hypos)
        else:
            rouge = Rouge()
            return rouge.get_scores(hyps=hypos, refs=refs, avg=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        for i in range(args.num_epochs):
            train_epoch(sess, i)

            if args.task == 'mt':
                valid_scores = []
                test_scores = []
                for j in range(3):
                    valid_scores.append(valid_epoch(sess))
                    test_scores.append(test_epoch(sess))

                print('valid epoch', i, ':', valid_scores,
                      'avg:', sum(valid_scores) / 3.,
                      file=valid_test_log_file)
                print('test epoch', i, ':', test_scores,
                      'avg:', sum(test_scores) / 3.,
                      file=valid_test_log_file)
                print('=' * 100, file=valid_test_log_file)
                valid_test_log_file.flush()
            else:
                valid_scores = valid_epoch(sess)
                test_scores = test_epoch(sess)

                print('valid epoch', i, ':', file=valid_test_log_file)
                for key, value in valid_scores.items():
                    print(key, value, file=valid_test_log_file)
                print('test epoch', i, ':', file=valid_test_log_file)
                for key, value in test_scores.items():
                    print(key, value, file=valid_test_log_file)
                print('=' * 100, file=valid_test_log_file)
                valid_test_log_file.flush()


if __name__ == '__main__':
    main()
