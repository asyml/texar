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
"""Example of building a sentence classifier based on pre-trained BERT model.
"""

import os
import importlib
import tensorflow as tf
import texar.tf as tx

from utils import model_utils

# pylint: disable=invalid-name, too-many-locals, too-many-statements

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "config_downstream", "config_classifier",
    "Configuration of the downstream part of the model.")
flags.DEFINE_string(
    "pretrained_model_name", 'bert-base-uncased',
    "The name of pre-trained BERT model. See the doc of "
    "`texar.tf.modules.PretrainedBERTMixin for all supported models.`")
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

    # Loads data
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
    input_length = tf.reduce_sum(1 - tf.cast(tf.equal(input_ids, 0), tf.int32),
                                 axis=1)
    # Builds BERT
    hparams = {
        'clas_strategy': 'cls_time'
    }
    model = tx.modules.BERTClassifier(
        pretrained_model_name=FLAGS.pretrained_model_name,
        hparams=hparams)
    logits, preds = model(input_ids, input_length, segment_ids)

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
    lr = model_utils.get_lr(global_step, num_train_steps,  # lr is a Tensor
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
                    tf.logging.info('step:%d; loss:%f;' % (step, rets['loss']))

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
