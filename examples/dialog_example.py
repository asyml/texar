#
"""
Example pipeline. This is a minimal example of basic RNN language model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-name-in-module

import os
import numpy as np
import tensorflow as tf
import texar as tx

from texar.modules.encoders.hierarchical_encoders_new import HierarchicalRNNEncoder
from texar.modules.decoders.beam_search_decode import beam_search_decode

from tensorflow.contrib.seq2seq import tile_batch

from argparse import ArgumentParser

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

parser = ArgumentParser()
parser.add_argument('-l', '--load_path', default=None, type=str)
parser.add_argument('--stage', nargs='+', 
                    default=['train', 'val', 'test'], type=str)
parser.add_argument('--test_batch_num', default=None, type=int)
parser.add_argument('--data_root', default='../data/dialog/', type=str)
parser.add_argument('--save_root', default='/tmp', type=str)
args = parser.parse_args()

data_hparams = {
    stage: {
        "num_epochs": 1,
        "batch_size": 30,
        "source_dataset": {
            "variable_utterance": True,
            "max_utterance_cnt": 2, 
            "files": [
                os.path.join(args.data_root, 'source.txt')],
            "vocab_file": os.path.join(args.data_root, 'vocab.txt'),
            "bos_token": tx.data.SpecialTokens.BOS
        },
        "target_dataset": {
            "files": [
                os.path.join(args.data_root, 'target.txt')],
            "vocab_share": True,
            "processing_share": True,
            "embedding_init_share": True,
        }
    }
    for stage in ['train', 'val', 'test']
}

encoder_minor_hparams = {
    "rnn_cell_fw": {
        "type": "GRUCell",
        "kwargs": {
            "num_units": 300,
            "kernel_initializer": tf.orthogonal_initializer(),
        },
        "dropout": {
            "input_keep_prob": 0.5,
        }
    },
    "rnn_cell_share_config": True
}
encoder_major_hparams = {
    "rnn_cell": {
        "type": "GRUCell",
        "kwargs": {
            "num_units": 600,
            "kernel_initializer": tf.orthogonal_initializer(),
        },
        "dropout": {
            "input_keep_prob": 0.5,
        }
    }
}
decoder_hparams = {
    "rnn_cell": {
        "type": "GRUCell",
        "kwargs": {
            "num_units": 400,
            "kernel_initializer": tf.orthogonal_initializer(),
        },
        "dropout": {
            "output_keep_prob": 0.5,
        }
    }
}

opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.001,
        }
    }
}

def main():
    train_data = tx.data.PairedTextData(data_hparams['train'])
    val_data = tx.data.PairedTextData(data_hparams['val'])
    test_data = tx.data.PairedTextData(data_hparams['test'])
    iterator = tx.data.TrainTestDataIterator(train=train_data,
                                             val=val_data,
                                             test=test_data)

    # declare modules
    embedder = tx.modules.WordEmbedder(
        vocab_size=train_data.source_vocab.size, hparams={'dim':200})

    encoder_minor = tx.modules.BidirectionalRNNEncoder(
        hparams=encoder_minor_hparams)
    encoder_major = tx.modules.UnidirectionalRNNEncoder(
        hparams=encoder_major_hparams)
    encoder = HierarchicalRNNEncoder(
        encoder_major, encoder_minor)

    decoder = tx.modules.BasicRNNDecoder(
        hparams=decoder_hparams, vocab_size=train_data.source_vocab.size)

    #connector = tf.layers.Dense(decoder.cell.state_size)
    connector = tx.modules.connectors.MLPTransformConnector(
        decoder.cell.state_size)

    # build graph
    data_batch = iterator.get_next()

    dialog_embed = embedder(data_batch['source_text_ids'])

    ecdr_states = encoder(
        dialog_embed, 
        sequence_length=data_batch['source_length'],
        sequence_length_major=data_batch['source_utterance_cnt'])[1]

    dcdr_states = connector(ecdr_states)

    # train branch
    outputs, _, lengths = decoder(
        initial_state=dcdr_states, 
        inputs=data_batch['target_text_ids'],
        sequence_length=data_batch['target_length'] - 1,
        embedding=embedder)

    mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=data_batch['target_text_ids'][:, 1:],
        logits=outputs.logits,
        sequence_length=lengths,
        sum_over_timesteps=False,
        average_across_timesteps=True)

    global_step = tf.Variable(0, name='global_step', trainable=True)
    train_op = tx.core.get_train_op(
        mle_loss, global_step=global_step, hparams=opt_hparams)

    # test branch
    
    test_batch_size = test_data.hparams.batch_size

    outputs_sample = decoder(
        decoding_strategy="infer_greedy",
        initial_state=dcdr_states,
        max_decoding_length=50,
        start_tokens=tf.cast(tf.fill(
            tf.shape(dcdr_states)[:1], train_data.source_vocab.bos_token_id),
            tf.int32),
        end_token=train_data.source_vocab.eos_token_id,
        embedding=embedder)[0]

    sample_text = train_data.source_vocab.map_ids_to_tokens(
        outputs_sample.sample_id)

    beam_search_samples, beam_states = beam_search_decode(
        decoder, 
        initial_state=tile_batch(dcdr_states, 5),
        max_decoding_length=50,
        start_tokens=tf.cast(tf.fill(
            tf.shape(dcdr_states)[:1], train_data.source_vocab.bos_token_id),
            tf.int32),
        end_token=train_data.source_vocab.eos_token_id,
        embedding=embedder,
        beam_width=5
    )
    beam_lengths = beam_states.lengths

    # denumericalize the generated samples
    beam_sample_text = train_data.source_vocab.map_ids_to_tokens(
        beam_search_samples.predicted_ids)

    target_tuple = (data_batch['target_text'][:, 1:], 
                    data_batch['target_length'] - 1)
    #train_data.source_vocab.map_ids_to_tokens(
    #data_batch['target_text_ids'][:, 1:]), 
    #data_batch['target_length'] - 1)

    dialog_tuple = (data_batch['source_text'], data_batch['source_length'],
                    data_batch['source_utterance_cnt'])

    def _train_epochs(sess, epoch, display=10):
        iterator.switch_to_train_data(sess)
        while True:
            try:
                feed = {tx.global_mode(): tf.estimator.ModeKeys.TRAIN}
                step, loss, _ = sess.run(
                    [global_step, mle_loss, train_op], feed_dict=feed)
                
                if step % display == 0:
                    print('step {} at epoch {}: loss={}'.format(
                        step, epoch, loss))

            except tf.errors.OutOfRangeError:
                print('epoch {} fin: loss={}'.format(epoch, loss))
                break 

    def _val_epochs(sess, epoch, loss_histories):
        iterator.switch_to_val_data(sess)

        valid_loss = []
        cnt = 0
        while True:
            try:
                feed = {tx.global_mode(): tf.estimator.ModeKeys.EVAL}
                loss = sess.run(mle_loss, feed_dict=feed)
                valid_loss.append(loss)

            except tf.errors.OutOfRangeError:
                loss = np.mean(valid_loss)
                print('epoch {} fin: loss={}'.format(epoch, loss))
                break 

        loss_histories.append(loss)
        best = min(loss_histories)
        
        return len(loss_histories) - loss_histories.index(best) - 1

    def _test_epochs(sess, epoch, test_batch_num=None):
        iterator.switch_to_test_data(sess)

        max_bleus = []
        avg_bleus = []
        txt_results = []
        
        batch_cnt = 0


        while batch_cnt != test_batch_num:
            try: 
                feed = {tx.global_mode(): tf.estimator.ModeKeys.EVAL}

                p = sess.run(data_batch, feed_dict=feed)

                #samples = sess.run(sample_text, feed_dict=feed)
                samples, lengths, dialog_t, target_t = sess.run(
                    [beam_sample_text, beam_lengths, 
                     dialog_tuple, target_tuple],
                    feed_dict=feed)

                for (beam, beam_len, 
                     dialog, utts_len, utts_cnt, 
                     target, tgt_len) in zip(
                    samples, lengths, *dialog_t, *target_t):

                    srcs = [dialog[i, :utts_len[i]] for i in range(utts_cnt)]

                    hyps = [beam[:l-1, i] for i, l in enumerate(beam_len)]
                    refs = [target[:tgt_len-1]]

                    scrs = [sentence_bleu(
                        refs, hyp, 
                        smoothing_function=SmoothingFunction().method7,
                        weights=[1.]) 
                        for hyp in hyps]

                    max_bleu, avg_bleu = np.max(scrs), np.mean(scrs)

                    max_bleus.append(max_bleu)
                    avg_bleus.append(avg_bleu)

                    src_txt = b'\n'.join([b' '.join(s[1:-1]) for s in srcs])
                    hyp_txt = b'\n'.join([b' '.join(s) for s in hyps])
                    ref_txt = b'\n'.join([b' '.join(s) for s in refs])
                    txt_results.append('input:\n{}\nhyps:\n{}\nref:\n{}'.format(
                        src_txt.decode(), hyp_txt.decode(), ref_txt.decode()))

            except tf.errors.OutOfRangeError:
                break

            batch_cnt += 1
            print('test batch {}/{}'.format(
                batch_cnt, test_batch_num), end='\r')
        
        bleu_recall = np.mean(max_bleus)
        bleu_prec = np.mean(avg_bleus)
                
        print('test epoch {}: bleu_recall={}, bleu_pred={}'.format(
            epoch, bleu_recall, bleu_prec))

        with open('test_txt_results.txt', 'w') as f:
            f.write('\n\n'.join(txt_results))


    saver = tf.train.Saver()
    with tf.Session() as sess:    
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        coord = tf.train.Coordinator()  
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if args.load_path:
            saver.restore(sess, args.load_path)

        loss_histories = []

        for epoch in range(100):
            if 'train' in args.stage:
                assert 'val' in args.stage
                _train_epochs(sess, epoch)
            if 'val' in args.stage:
                best_index_diff = _val_epochs(sess, epoch, loss_histories)
            if 'test' in args.stage:
                _test_epochs(sess, epoch, args.test_batch_num) 
        
            if 'train' in args.stage:
                if best_index_diff == 0:
                    saver.save(sess, '/tmp/hierarchical_example_best.ckpt')
                elif best_index_diff > 5:
                    print('overfit at epoch {}'.format(epoch))
                    break
            else:
                break
            
if __name__ == "__main__":
    main()
