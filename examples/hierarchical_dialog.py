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
from texar.modules import BasicRNNDecoder, get_helper
from texar.modules import HierarchicalEncoder
from texar.losses import mle_losses
from texar.core import optimization as opt
from texar import context
from texar.data.embedding import Embedding
from texar.modules.embedders.embedders import WordEmbedder

from IPython import embed

if __name__ == "__main__":
    ### Build data pipeline

    # Config data hyperparams. Hyperparams not configured will be automatically
    # filled with default values. For text database, default values are defined
    # in `texar.data.database.default_text_dataset_hparams()`.
    # Construct database
    
    data_hparams = {
        "num_epochs": 10,
        "batch_size": 10,
        "source_dataset": {
            "files": ['../data/dialog/source.txt'],
            "vocab_file": '../data/dialog/vocab.txt',
            "processing": {
                "max_seq_length": 30,
                "max_context_length": 2 
            },
            "embedding_init": {
                "file": '/space/hzt/word_embed/glove/glove.6B.200d.txt',
                "dim": 200,
                "read_fn": "load_glove"
            }
        },
        "target_dataset": {
            "files": ['../data/dialog/target.txt'],
            "vocab_share": True,
            "reader_share": True,
        }
    }

    # Construct the database
    dialog_db = qMultiSourceTextData(data_hparams)
    data_batch = dialog_db()

    # build embedder

    embedder = WordEmbedder(init_value=dialog_db.embedding.word_vecs)

    # builder encoder
    encoder = HierarchicalEncoder()

    # Build decoder. Simply use the default hyperparameters.\
    decoder = BasicRNNDecoder(vocab_size=dialog_db.target_vocab.vocab_size)

    # Build connector, which simply feeds zero state to decoder as initial state
    connector = ForwardConnector(decoder.state_size)
    
    inputs_embedded = embedder(inputs=data_batch['source_text_ids'])
    enc_outputs, enc_last = encoder(inputs=inputs_embedded)

    # We shall probably improve the interface here.
    helper_train = get_helper(
        "EmbeddingTrainingHelper",
        inputs=data_batch['target_text_ids'][:, :-1],
        sequence_length=data_batch['target_length'] - 1,
        embedding=embedder)

    embed()

    # Decode
    outputs, final_state, sequence_lengths = decoder(
        helper=helper_train, initial_state=connector(enc_last))

    embed()

    # Build loss
    mle_loss = mle_losses.average_sequence_sparse_softmax_cross_entropy(
        labels=data_batch['target_text_ids'][:, 1:],
        logits=outputs.logits,
        sequence_length=sequence_lengths - 1)

    # Build train op. Only config the optimizer while using default settings
    # for other hyperparameters.
    opt_hparams = {
        "optimizer": {
            "type": "MomentumOptimizer",
            "kwargs": {
                "learning_rate": 0.01,
                "momentum": 0.9
            }
        }
    }
    train_op, global_step = opt.get_train_op(mle_loss, hparams=opt_hparams)

    ### Graph is done. Now start running
    # We shall wrap these environment setup codes
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                # Run the logics
                _, step, loss = sess.run(
                    [train_op, global_step, mle_loss],
                    feed_dict={context.is_train(): True})

                if step % 10 == 0:
                    print("%d: %.6f" % (step, loss))

        except tf.errors.OutOfRangeError:
            print('Done -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

