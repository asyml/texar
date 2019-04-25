# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example of building a sentence classifier based on pre-trained BERT
model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import importlib
import tensorflow as tf
import texar as tx

from utils import model_utils

# pylint: disable=invalid-name, too-many-locals, too-many-statements

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "config_bert_pretrain", 'uncased_L-12_H-768_A-12',
    "The architecture of pre-trained BERT model to use.")
flags.DEFINE_string(
    "config_format_bert", "json",
    "The configuration format. Set to 'json' if the BERT config file is in "
    "the same format of the official BERT config file. Set to 'texar' if the "
    "BERT config file is in Texar format.")
flags.DEFINE_string(
    "config_downstream", "config_classifier",
    "Configuration of the downstream part of the model and optmization.")
flags.DEFINE_string(
    "config_data", "config_data",
    "The dataset config.")
flags.DEFINE_string(
    "output_dir", "output/",
    "The output directory where the model checkpoints will be written.")
flags.DEFINE_string(
    "checkpoint", None,
    "Path to a model checkpoint (including bert modules) to restore from.")
flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_test", False, "Whether to run test on the test set.")
flags.DEFINE_bool("distributed", False, "Whether to run in distributed mode.")

config_data = importlib.import_module(FLAGS.config_data)
config_downstream = importlib.import_module(FLAGS.config_downstream)

def main(_):
    """
    Builds the model and runs.
    """

    if FLAGS.distributed:
        import horovod.tensorflow as hvd
        hvd.init()

    tf.logging.set_verbosity(tf.logging.INFO)

    tx.utils.maybe_create_dir(FLAGS.output_dir)

    bert_pretrain_dir = ('bert_pretrained_models'
                         '/%s') % FLAGS.config_bert_pretrain
    # Loads BERT model configuration
    if FLAGS.config_format_bert == "json":
        bert_config = model_utils.transform_bert_to_texar_config(
            os.path.join(bert_pretrain_dir, 'bert_config.json'))
    elif FLAGS.config_format_bert == 'texar':
        bert_config = importlib.import_module(
            ('bert_config_lib.'
             'config_model_%s') % FLAGS.config_bert_pretrain)
    else:
        raise ValueError('Unknown config_format_bert.')

    # Loads data

    num_classes = config_data.num_classes
    num_train_data = config_data.num_train_data

    # Configures distribued mode
    if FLAGS.distributed:
        config_data.train_hparam["dataset"]["num_shards"] = hvd.size()
        config_data.train_hparam["dataset"]["shard_id"] = hvd.rank()
        config_data.train_hparam["batch_size"] //= hvd.size()

    train_dataset = tx.data.TFRecordData(hparams=config_data.train_hparam)
    eval_dataset = tx.data.TFRecordData(hparams=config_data.eval_hparam)
    test_dataset = tx.data.TFRecordData(hparams=config_data.test_hparam)

    iterator = tx.data.FeedableDataIterator({
        'train': train_dataset, 'eval': eval_dataset, 'test': test_dataset})
    batch = iterator.get_next()
    input_ids = batch["input_ids"]
    segment_ids = batch["segment_ids"]
    batch_size = tf.shape(input_ids)[0]
    input_length = tf.reduce_sum(1 - tf.to_int32(tf.equal(input_ids, 0)),
                                 axis=1)
    # Builds BERT
    with tf.variable_scope('bert'):
        # Word embedding
        embedder = tx.modules.WordEmbedder(
            vocab_size=bert_config.vocab_size,
            hparams=bert_config.embed)
        word_embeds = embedder(input_ids)

        # Segment embedding for each type of tokens
        segment_embedder = tx.modules.WordEmbedder(
            vocab_size=bert_config.type_vocab_size,
            hparams=bert_config.segment_embed)
        segment_embeds = segment_embedder(segment_ids)

        # Position embedding
        position_embedder = tx.modules.PositionEmbedder(
            position_size=bert_config.position_size,
            hparams=bert_config.position_embed)
        seq_length = tf.ones([batch_size], tf.int32) * tf.shape(input_ids)[1]
        pos_embeds = position_embedder(sequence_length=seq_length)

        # Aggregates embeddings
        input_embeds = word_embeds + segment_embeds + pos_embeds

        # The BERT model (a TransformerEncoder)
        encoder = tx.modules.TransformerEncoder(hparams=bert_config.encoder)
        output = encoder(input_embeds, input_length)

        # Builds layers for downstream classification, which is also
        # initialized with BERT pre-trained checkpoint.
        with tf.variable_scope("pooler"):
            # Uses the projection of the 1st-step hidden vector of BERT output
            # as the representation of the sentence
            bert_sent_hidden = tf.squeeze(output[:, 0:1, :], axis=1)
            bert_sent_output = tf.layers.dense(
                bert_sent_hidden, config_downstream.hidden_dim,
                activation=tf.tanh)
            output = tf.layers.dropout(
                bert_sent_output, rate=0.1, training=tx.global_mode_train())

    # Adds the final classification layer
    logits = tf.layers.dense(
        output, num_classes,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    accu = tx.evals.accuracy(batch['label_ids'], preds)

    # Optimization

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=batch["label_ids"], logits=logits)
    global_step = tf.Variable(0, trainable=False)

    # Builds learning rate decay scheduler
    static_lr = config_downstream.lr['static_lr']
    num_train_steps = int(num_train_data / config_data.train_batch_size
                          * config_data.max_train_epoch)
    num_warmup_steps = int(num_train_steps * config_data.warmup_proportion)
    lr = model_utils.get_lr(global_step, num_train_steps, # lr is a Tensor
                            num_warmup_steps, static_lr)

    opt = tx.core.get_optimizer(
        global_step=global_step,
        learning_rate=lr,
        hparams=config_downstream.opt
    )

    if FLAGS.distributed:
        opt = hvd.DistributedOptimizer(opt)

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=None,
        optimizer=opt)

    # Train/eval/test routine

    def _is_head():
        if not FLAGS.distributed:
            return True
        else:
            return hvd.rank() == 0

    def _train_epoch(sess):
        """Trains on the training set, and evaluates on the dev set
        periodically.
        """
        iterator.restart_dataset(sess, 'train')
        
        fetches = {
            'train_op': train_op,
            'loss': loss,
            'batch_size': batch_size,
            'step': global_step
        }

        while True:
            try:
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, 'train'),
                    tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
                }
                rets = sess.run(fetches, feed_dict)
                step = rets['step']

                dis_steps = config_data.display_steps
                if _is_head() and dis_steps > 0 and step % dis_steps == 0:
                    tf.logging.info('step:%d; loss:%f' % (step, rets['loss']))

                eval_steps = config_data.eval_steps
                if _is_head() and eval_steps > 0 and step % eval_steps == 0:
                    _eval_epoch(sess)

            except tf.errors.OutOfRangeError:
                break

    def _eval_epoch(sess):
        """Evaluates on the dev set.
        """
        iterator.restart_dataset(sess, 'eval')

        cum_acc = 0.0
        cum_loss = 0.0
        nsamples = 0
        fetches = {
            'accu': accu,
            'loss': loss,
            'batch_size': batch_size,
        }
        while True:
            try:
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, 'eval'),
                    tx.context.global_mode(): tf.estimator.ModeKeys.EVAL,
                }
                rets = sess.run(fetches, feed_dict)

                cum_acc += rets['accu'] * rets['batch_size']
                cum_loss += rets['loss'] * rets['batch_size']
                nsamples += rets['batch_size']
            except tf.errors.OutOfRangeError:
                break

        tf.logging.info('eval accu: {}; loss: {}; nsamples: {}'.format(
            cum_acc / nsamples, cum_loss / nsamples, nsamples))

    def _test_epoch(sess):
        """Does predictions on the test set.
        """
        iterator.restart_dataset(sess, 'test')

        _all_preds = []
        while True:
            try:
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, 'test'),
                    tx.context.global_mode(): tf.estimator.ModeKeys.PREDICT,
                }
                _preds = sess.run(preds, feed_dict=feed_dict)
                _all_preds.extend(_preds.tolist())
            except tf.errors.OutOfRangeError:
                break

        output_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_file, "w") as writer:
            writer.write('\n'.join(str(p) for p in _all_preds))

    # Loads pretrained BERT model parameters
    init_checkpoint = os.path.join(bert_pretrain_dir, 'bert_model.ckpt')
    model_utils.init_bert_checkpoint(init_checkpoint)

    # Broadcasts global variables from rank-0 process
    if FLAGS.distributed:
        bcast = hvd.broadcast_global_variables(0)

    session_config = tf.ConfigProto()
    if FLAGS.distributed:
        session_config.gpu_options.visible_device_list = str(hvd.local_rank())

    with tf.Session(config=session_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        if FLAGS.distributed:
            bcast.run()

        # Restores trained model if specified
        saver = tf.train.Saver()
        if FLAGS.checkpoint:
            saver.restore(sess, FLAGS.checkpoint)

        iterator.initialize_dataset(sess)

        if FLAGS.do_train:
            for i in range(config_data.max_train_epoch):
                _train_epoch(sess)
            saver.save(sess, FLAGS.output_dir + '/model.ckpt')

        if FLAGS.do_eval:
            _eval_epoch(sess)

        if FLAGS.do_test:
            _test_epoch(sess)

if __name__ == "__main__":
    tf.app.run()
