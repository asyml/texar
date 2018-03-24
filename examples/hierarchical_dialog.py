#
"""
Example pipeline. This is a minimal example of basic RNN language model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-name-in-module

import tensorflow as tf

# We shall wrap all these modules
from texar.data import qMultiSourceTextData
from texar.modules import ForwardConnector
from texar.modules import BasicRNNDecoder, SoftmaxEmbeddingHelper, get_helper
from texar.modules import HierarchicalRNNEncoder, UnidirectionalRNNEncoder, \
                          BidirectionalRNNEncoder
from texar.losses import mle_losses
from texar.core import optimization as opt
from texar.modules import WordEmbedder
from texar import context

import texar as tx

import os
from argparse import ArgumentParser
from copy import deepcopy

from IPython import embed

parser = ArgumentParser()
parser.add_argument('mode', choices=['train', 'test'])
parser.add_argument('-s', '--save_root', required=True, type=str)
parser.add_argument('-l', '--load_path', type=str, default=None)
parser.add_argument('-n', '--num_epoch', type=int, default=None)

args = parser.parse_args()

data_root = "/home/hzt/yxj/hred-sw1r2c"

train_data_hparams = {
    "seed": 173,
    "batch_size": 10,
    "num_epochs": 1,
    "source_dataset": {
        "variable_utterance": True,
        "max_utterance_cnt": 10,
        "max_seq_length": None,
        "embedding_init": {
            "file": "/space/hzt/word_embed/glove/glove.6B.200d.txt",
            "dim": 200,
            "read_fn": "load_glove",
        },
        "files": [os.path.join(data_root, 'train-source.txt')],
        "vocab_file": os.path.join(data_root, 'vocab.txt'),
    },
    "target_dataset": {
        "files": [os.path.join(data_root, 'train-target.txt')],
        "vocab_share": True,
        "embedding_init_share": True,
    }
}
opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
    }
}

valid_data_hparams = deepcopy(train_data_hparams)
test_data_hparams  = deepcopy(train_data_hparams)

valid_data_hparams['source_dataset']['files'] = [os.path.join(
                                                    data_root, 
                                                    'valid-source.txt'
                                                 )]
valid_data_hparams['target_dataset']['files'] = [os.path.join(
                                                    data_root,
                                                    'valid-target.txt'
                                                 )]
test_data_hparams['source_dataset']['files'] = [os.path.join(
                                                    data_root, 
                                                    'test-source.txt'
                                                )]
test_data_hparams['target_dataset']['files'] = [os.path.join(
                                                    data_root,
                                                    'test-target.txt'
                                                )]
# declaration of modules

train_data = tx.data.PairedTextData(train_data_hparams)
valid_data = tx.data.PairedTextData(valid_data_hparams)
test_data  = tx.data.PairedTextData(test_data_hparams)

iterator = tx.data.TrainTestDataIterator(train=train_data,
                                         val=valid_data,
                                         test=test_data)

embedder = WordEmbedder(init_value=train_data.source_embedding_init_value) 

encoder_major = UnidirectionalRNNEncoder()
encoder_minor = BidirectionalRNNEncoder()
    
encoder = HierarchicalRNNEncoder(encoder_major=encoder_major,
                                 encoder_minor=encoder_minor)

decoder = BasicRNNDecoder(vocab_size=train_data.source_vocab.size)
    
connector = ForwardConnector(decoder.state_size)
    
# build the graph

data_batch = iterator.get_next()

inputs_embedded = embedder(inputs=data_batch['source_text_ids'])

encoder_outputs, encoder_final = encoder(inputs=inputs_embedded,
                                         sequence_length_minor=
                                            data_batch['source_length'])

helper_train = get_helper(
    "EmbeddingTrainingHelper",
    inputs=data_batch['target_text_ids'][:, :-1],
    sequence_length=data_batch['target_length'] - 1,
    embedding=embedder)

outputs, final_states, sequence_length = decoder(
    helper=helper_train,
    initial_state=connector(encoder_final))
    
mle_loss = mle_losses.average_sequence_sparse_softmax_cross_entropy(
    labels=data_batch['target_text_ids'][:, 1:],
    logits=outputs.logits,
    sequence_length=sequence_length - 1)

train_op, global_step = opt.get_train_op(mle_loss, hparams=opt_hparams) 

def train(sess, epoch_cnt):
    iterator.switch_to_train_data(sess)

    while True:
        try:
            _, step, loss = sess.run(
                [train_op, global_step, mle_loss],
                feed_dict={context.global_mode(): True})
            if step % 10 == 0:
                print("step %d, at epoch %d: %.6f" % (step, epoch_cnt, 
                                                      loss))

        except tf.errors.OutOfRangeError:
            break

def valid(sess, epoch_cnt):
    iterator.switch_to_val_data(sess)

    item_cnt = 0
    loss_cnt = 0
    while True:
        try:
            f_output, loss = sess.run(
                [outputs, mle_loss],
                feed_dict={context.global_mode(): False})

            item_cnt += len(f_output.logits)
            loss_cnt += len(f_output.logits) * loss  
             
        except tf.errors.OutOfRangeError:
            break

    print("valid at epoch %d: %.6f" % (epoch_cnt, loss_cnt / item_cnt))

    return loss_cnt / item_cnt

def test(sess):
    iterator.switch_to_test_data(sess)

    item_cnt = 0
    loss_cnt = 0
    while True:
        try:
            f_output, loss = sess.run(
                [outputs, mle_loss],
                feed_dict={context.global_mode(): False})

            item_cnt += len(f_output.logits)
            loss_cnt += len(f_output.logits) * loss  

            print(item_cnt, loss)
             
        except tf.errors.OutOfRangeError:
            break

    print("test: %.6f" % (loss_cnt / item_cnt))

if __name__ == "__main__":

    saver = tf.train.Saver()

    best_valid_loss = 1e10
    best_epoch_loc = None

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())


        epoch_cnt = 0

        if args.load_path is not None:
            saver.restore(sess, args.load_path)

        if args.mode == 'test':
            test(sess)
        else:
            while epoch_cnt != args.num_epoch: 
                epoch_cnt += 1

                train(sess, epoch_cnt)
                valid_loss = valid(sess, epoch_cnt)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch_loc = epoch_cnt
                    saver.save(sess, args.save_root)              
            
                if epoch_cnt - best_epoch_loc > 10:
                    print('seems overfit after {}. HALT.'.format(
                            best_epoch_loc))
                    break


