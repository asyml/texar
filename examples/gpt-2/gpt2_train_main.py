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
"""Example of fine-tuning OpenAI GPT-2 language model.
"""

import os
import importlib
import tensorflow as tf
import texar.tf as tx

from utils import model_utils, processor

# pylint: disable=invalid-name, too-many-locals, too-many-statements, no-member
# pylint: disable=too-many-branches

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint", None,
                    "Model checkpoint to resume training or for test.")
flags.DEFINE_string("pretrain_checkpoint",
                    "gpt2_pretrained_models/model_117M/model.ckpt",
                    "OpenAI pretrained model checkpoint. Ignored if "
                    "'--checkpoint' is specified.")
flags.DEFINE_string("pretrain_model_dir", "gpt2_pretrained_models/model_117M",
                    "The directory of pretrained model, for loading vocabuary, "
                    "etc.")
flags.DEFINE_float("temperature", 0.7,
                   "Softmax temperature for top-k sample decoding. Must be "
                   "strictly greater than 0. Defaults to 0.7.")
flags.DEFINE_integer("top_k", 40,
                     "The number of top most likely candidates from a vocab "
                     "distribution.")
flags.DEFINE_string("config_train", "configs.config_train",
                    "Configurations of GPT-2 training, including data and "
                    "optimization hyperparameters.")
flags.DEFINE_string("config_type", "texar",
                    "The configuration file type. Set to 'json' if the GPT-2 "
                    "config file is in the same type of the official GPT-2 "
                    "config file. Set to 'texar' if GPT-2 config file is in "
                    "Texar type.")
flags.DEFINE_string("config_model", "configs.config_model_117M",
                    "The model configuration file to configure the model. "
                    "The config file type is define by the 'config_type',"
                    "it be of texar type or json type."
                    "For '--config_type=json', set the json config file path"
                    "like: '--config_model gpt2_pretrained_models/model_117M/"
                    "hparams.json';"
                    "For '--config_type=texar', set the texar config file "
                    "like: '--config_model configs.config_model_117M'.")
flags.DEFINE_string("output_dir", "output/",
                    "The output directory where the model checkpoints will be "
                    "written.")
flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_test", False, "Whether to run test on the test set.")
flags.DEFINE_bool("distributed", False, "Whether to run in distributed mode.")

config_train = importlib.import_module(FLAGS.config_train)


def main(_):
    """
    Builds the model and runs
    """
    if FLAGS.distributed:
        import horovod.tensorflow as hvd
        hvd.init()

    tf.logging.set_verbosity(tf.logging.INFO)

    # Loads GPT-2 model configuration

    if FLAGS.config_type == "json":
        gpt2_config = model_utils.transform_gpt2_to_texar_config(
            FLAGS.config_model)
    elif FLAGS.config_type == 'texar':
        gpt2_config = importlib.import_module(
            FLAGS.config_model)
    else:
        raise ValueError('Unknown config_type.')

    # Creates a data pre-processor for, e.g., BPE encoding
    proc = processor.get_encoder(FLAGS.pretrain_model_dir)

    max_decoding_length = config_train.max_decoding_length
    assert max_decoding_length <= gpt2_config.position_size, (
        "max_decoding_length should not be greater than position_size. "
        "{}>{}".format(max_decoding_length, gpt2_config.position_size))

    # Loads data

    # Configures training data shard in distributed mode
    if FLAGS.distributed:
        config_train.train_hparam["dataset"]["num_shards"] = hvd.size()
        config_train.train_hparam["dataset"]["shard_id"] = hvd.rank()
        config_train.train_hparam["batch_size"] //= hvd.size()

    datasets = {}
    if FLAGS.do_train:
        train_dataset = tx.data.TFRecordData(hparams=config_train.train_hparam)
        datasets['train'] = train_dataset
    if FLAGS.do_eval:
        dev_dataset = tx.data.TFRecordData(hparams=config_train.dev_hparam)
        datasets['dev'] = dev_dataset
    if FLAGS.do_test:
        test_dataset = tx.data.TFRecordData(hparams=config_train.test_hparam)
        datasets['test'] = test_dataset
    iterator = tx.data.FeedableDataIterator(datasets)
    batch = iterator.get_next()
    batch_size = tf.shape(batch['text_ids'])[0]

    # Builds the GPT-2 model

    word_embedder = tx.modules.WordEmbedder(
        vocab_size=gpt2_config.vocab_size,
        hparams=gpt2_config.embed)

    pos_embedder = tx.modules.PositionEmbedder(
        position_size=gpt2_config.position_size,
        hparams=gpt2_config.pos_embed)

    # Ties output layer with input word embedding
    output_layer = tf.transpose(word_embedder.embedding, (1, 0))

    decoder = tx.modules.TransformerDecoder(
        vocab_size=gpt2_config.vocab_size,
        output_layer=output_layer,
        hparams=gpt2_config.decoder)

    # For training
    seq_len = tf.fill([batch_size], tf.shape(batch['text_ids'])[1])
    pos_embeds = pos_embedder(sequence_length=seq_len)
    input_embeds = word_embedder(batch['text_ids']) + pos_embeds

    outputs = decoder(inputs=input_embeds, decoding_strategy='train_greedy')

    loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=batch['text_ids'][:, 1:],
        logits=outputs.logits[:, :-1, :],
        sequence_length=batch['length'] - 1,
        average_across_timesteps=True,
        sum_over_timesteps=False)
    ppl = tf.exp(loss)

    global_step = tf.Variable(0, trainable=False)
    opt = tx.core.get_optimizer(
        global_step=global_step,
        hparams=config_train.opt)

    if FLAGS.distributed:
        opt = hvd.DistributedOptimizer(opt)

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=None,
        optimizer=opt)

    # For generation: generates continuations of test text
    def _embedding_fn(x, y):
        # `x` is token ids, `y` is time steps
        return word_embedder(x) + pos_embedder(y)

    end_token = proc.encoder['<|endoftext|>']
    start_tokens = batch['text_ids'][:, 0]
    helper = tx.modules.TopKSampleEmbeddingHelper(
        embedding=_embedding_fn,
        start_tokens=start_tokens,
        end_token=end_token,
        top_k=FLAGS.top_k,
        softmax_temperature=FLAGS.temperature)

    outputs_infer, _ = decoder(
        context=batch['text_ids'],
        context_sequence_length=batch['length'],
        max_decoding_length=max_decoding_length,
        helper=helper)
    sample_id = outputs_infer.sample_id

    # Train/eval/test routine
    saver = tf.train.Saver()
    saver_best = tf.train.Saver(max_to_keep=1)
    dev_best = {'loss': 1e8, 'ppl': 1e8}

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
            'loss': train_op,
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

                dis_steps = config_train.display_steps
                if _is_head() and dis_steps > 0 and step % dis_steps == 0:
                    tf.logging.info('step:%d; loss:%f' % (step, rets['loss']))

                eval_steps = config_train.eval_steps
                if _is_head() and eval_steps > 0 and step % eval_steps == 0:
                    _dev_epoch(sess)

                ckpt_steps = config_train.checkpoint_steps
                if _is_head() and ckpt_steps > 0 and step % ckpt_steps == 0:
                    ckpt_fn = os.path.join(FLAGS.output_dir, 'model.ckpt')
                    ckpt_fn = saver.save(sess, ckpt_fn, global_step=step)
                    tf.logging.info('Checkpoint to {}'.format(ckpt_fn))

            except tf.errors.OutOfRangeError:
                break

    def _dev_epoch(sess):
        """Evaluates on the dev set.
        """
        iterator.restart_dataset(sess, 'dev')

        cum_loss = 0.
        cum_ppl = 0.
        nsamples = 0
        fetches = {
            'loss': loss,
            'ppl': ppl,
            'batch_size': batch_size,
        }
        while True:
            try:
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, 'dev'),
                    tx.context.global_mode(): tf.estimator.ModeKeys.EVAL,
                }
                rets = sess.run(fetches, feed_dict)

                cum_loss += rets['loss'] * rets['batch_size']
                cum_ppl += rets['ppl'] * rets['batch_size']
                nsamples += rets['batch_size']
            except tf.errors.OutOfRangeError:
                break

        avg_loss = cum_loss / nsamples
        avg_ppl = cum_ppl / nsamples
        tf.logging.info('dev loss: {}; ppl: {}; nsamples: {}'.format(
            avg_loss, avg_ppl, nsamples))

        if FLAGS.do_train and avg_loss < dev_best['loss']:
            dev_best['loss'] = avg_loss
            dev_best['ppl'] = avg_ppl
            ckpt_fn = os.path.join(FLAGS.output_dir, 'model_best.ckpt')
            ckpt_fn = saver_best.save(sess, ckpt_fn)
            tf.logging.info('Checkpoint best to {}'.format(ckpt_fn))

    def _test_epoch(sess):
        """Generates samples on the test set.
        """
        iterator.restart_dataset(sess, 'test')

        _all_inputs = []
        _all_samples = []
        fetches = {
            'inputs': batch['text_ids'],
            'length': batch['length'],
            'samples': sample_id
        }
        while True:
            try:
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, 'test'),
                    tx.context.global_mode(): tf.estimator.ModeKeys.PREDICT,
                }
                rets = sess.run(fetches, feed_dict=feed_dict)

                _inputs = []
                for i, l in zip(rets['inputs'], rets['length']):
                    # Delete padding
                    _inputs.append(i[:l].tolist())
                _all_inputs.extend(_inputs)

                _samples = []
                for s, l in zip(rets['samples'], rets['length']):
                    # Delete inputs from samples
                    _samples.append(s[l:].tolist())
                _all_samples.extend(_samples)

            except tf.errors.OutOfRangeError:
                break

        # Parse samples and write to file

        eos_token_id = proc.encoder['<|endoftext|>']

        _all_input_text = []
        for i in _all_inputs:
            if i[0] == eos_token_id:
                # '<|endoftext|>' is used as the BOS token. Delete it here
                i = i[1:]
            i_text = proc.decode(i)
            _all_input_text.append(i_text)
        # '<|endoftext|>' is used as the PAD token. Delete them here
        _all_input_text = tx.utils.strip_eos(_all_input_text,
                                             eos_token='<|endoftext|>')

        _all_samples_text = []
        for i, s in zip(_all_inputs, _all_samples):
            s_text = proc.decode(s)
            s_text = s_text.replace('\n', ' ')
            _all_samples_text.append(s_text)
        _all_samples_text = tx.utils.strip_eos(_all_samples_text,
                                               eos_token='<|endoftext|>')

        output_file = os.path.join(FLAGS.output_dir, "test_samples.tsv")
        tf.logging.info('Write samples to {}'.format(output_file))
        tx.utils.write_paired_text(
            _all_input_text, _all_samples_text, output_file)

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
        if FLAGS.checkpoint:
            tf.logging.info('Restore from {}'.format(FLAGS.checkpoint))
            saver.restore(sess, FLAGS.checkpoint)
        elif FLAGS.pretrain_checkpoint:
            tf.logging.info('Restore from {}'.format(FLAGS.pretrain_checkpoint))
            model_utils.init_gpt2_checkpoint(sess, FLAGS.pretrain_checkpoint)
            print("\nFinished loading\n")

        iterator.initialize_dataset(sess)

        if FLAGS.do_train:
            for _ in range(config_train.max_train_epoch):
                _train_epoch(sess)
            saver.save(sess, FLAGS.output_dir + '/model.ckpt')

        if FLAGS.do_eval:
            _dev_epoch(sess)

        if FLAGS.do_test:
            _test_epoch(sess)


if __name__ == "__main__":
    tf.app.run()
