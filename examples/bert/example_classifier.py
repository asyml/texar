# Modified based on Bert Official Release
# https://github.com/google-research/bert
"""A minimal example of fine-tuning BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tokenization
import importlib
import tensorflow as tf
import texar as tx
from texar.modules import TransformerEncoder, TransformerDecoder
from texar.utils import transformer_utils
from texar.utils.mode import is_train_mode
from texar.core import get_train_op
#from utils import get_train_op
from data.data_utils import MrpcProcessor,\
    file_based_convert_examples_to_features, file_based_input_fn_builder
import utils

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('config_data', 'config_mrpc', "The dataset config.")
flags.DEFINE_string(
    "bert_config_format", "texar",
    "The configuration format. Choose `json` if loaded from the config attached"
    "Choose `texar` to load the customed writen configuration file for texar")

flags.DEFINE_string(
    "bert_pretrain_config", 'uncased_L-12_H-768_A-12',
    "The config json file corresponding to the pre-trained BERT model.")

flags.DEFINE_string(
    "config_model", "config_model",
    "Model configuration for downstream task and the model training")
flags.DEFINE_float(
    "learning_rate", 2e-5,
    "Specify the model learining rate.")
flags.DEFINE_string(
    "saved_model", None,
    "The complete saved checkpoint (including bert modules), which can be restored from.")

flags.DEFINE_string(
    "output_dir", "output/",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_predict", False, "Whether to run predict on the test set.")

config_data = importlib.import_module(FLAGS.config_data)
config_model = importlib.import_module(FLAGS.config_model)
def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.gfile.MakeDirs(FLAGS.output_dir)

    # load BERT Model Configuration
    if FLAGS.bert_config_format == "json":
        bert_config = utils.transform_bert_to_texar_config(
            'bert_released_models/%s/bert_config.json'%FLAGS.bert_pretrain_config)
    elif FLAGS.bert_config_format == 'texar':
        bert_config = importlib.import_module(
            'bert_config_lib.config_model_%s' % (FLAGS.bert_pretrain_config))

    # Data Loading Configuration
    processor = MrpcProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    tokenizer = tokenization.FullTokenizer(
        vocab_file='bert_released_models/%s/vocab.txt'
            %(FLAGS.bert_pretrain_config),
        do_lower_case=FLAGS.do_lower_case)

    train_examples = processor.get_train_examples(config_data.data_dir)
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, label_list, config_data.max_seq_length,
        tokenizer, train_file)
    train_dataset = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=config_data.max_seq_length,
        drop_remainder=True,
        is_training=True)({'batch_size': config_data.train_batch_size})

    num_train_steps = int(len(train_examples) / config_data.train_batch_size \
        * config_data.max_train_epoch)
    num_warmup_steps = int(num_train_steps * config_data.warmup_proportion)
    eval_examples = processor.get_dev_examples(config_data.data_dir)
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, config_data.max_seq_length, tokenizer, eval_file)
    eval_dataset = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=config_data.max_seq_length,
        is_training=False,
        drop_remainder=False)({'batch_size': config_data.eval_batch_size})

    predict_examples = processor.get_test_examples(config_data.data_dir)
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(
        predict_examples, label_list,
        config_data.max_seq_length, tokenizer, predict_file)
    test_dataset = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=config_data.max_seq_length,
        is_training=False,
        drop_remainder=False)({'batch_size': config_data.test_batch_size})

    iterator = tx.data.FeedableDataIterator({
        'train': train_dataset,
        'eval': eval_dataset,
        'test': test_dataset})
    batch = iterator.get_next()
    input_ids = batch["input_ids"]
    segment_ids = batch["segment_ids"]

    batch_size = tf.shape(input_ids)[0]
    input_length = tf.reduce_sum(
        1 - tf.to_int32(tf.equal(input_ids, 0)), axis=1)

    # BERT (Transformer) model configuration
    mode = None # to follow the global mode
    with tf.variable_scope('bert'):
        embedder = tx.modules.WordEmbedder(
            vocab_size=bert_config.vocab_size,
            hparams=bert_config.embed)
        token_type_embedder = tx.modules.WordEmbedder(
            vocab_size=bert_config.type_vocab_size,
            hparams=bert_config.token_embed)
        word_embeds = embedder(input_ids, mode=mode)
        token_type_ids = segment_ids
        token_type_embeds = token_type_embedder(token_type_ids, mode=mode)
        input_embeds = word_embeds + token_type_embeds
        encoder = TransformerEncoder(hparams=bert_config.encoder)
        output_layer = encoder(input_embeds, input_length, mode=mode)

    # Downstream model configuration
        with tf.variable_scope("pooler"):
            # Use the projection of first token hidden vector of BERT output
            # as the representation of the sentence
            bert_sent_hidden = tf.squeeze(output_layer[:, 0:1, :], axis=1)
            bert_sent_output = tf.layers.dense(
                bert_sent_hidden, config_model.hidden_dim, activation=tf.tanh)
            output_layer = tf.layers.dropout(bert_sent_output, rate=0.1,
            training=is_train_mode(mode))

    logits = tf.layers.dense(output_layer, num_labels,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
    probabilities = tf.nn.softmax(logits, axis=-1)
    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)

    # Losses & train_ops
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=batch["label_ids"], logits=logits)
    #global_step = tf.train.get_or_create_global_step()
    global_step = tf.Variable(0, trainable=False)
    static_lr = config_model.lr['static_lr']
    lr = utils.get_lr(global_step, num_train_steps, num_warmup_steps, static_lr)
    train_op = get_train_op(
        loss,
        global_step=global_step,
        learning_rate=lr,
        hparams=config_model.opt)

    # Monitering data
    accu = tx.evals.accuracy(batch['label_ids'], preds)

    def _run_epoch(sess, mode):
        fetches = {
            'accu': accu,
            'batch_size': batch_size,
            'step': global_step,
            'loss': loss,
        }

        if mode == 'train':
            fetches['train_op'] = train_op
            while True:
                try:
                    feed_dict = {
                        iterator.handle: iterator.get_handle(sess, 'train'),
                        tx.context.global_mode(): tf.estimator.ModeKeys.TRAIN,
                    }
                    rets = sess.run(fetches, feed_dict)
                    #if rets['step'] % 50 == 0:
                    tf.logging.info('step:%d loss:%f' % (
                        rets['step'], rets['loss']))
                    if rets['step'] == num_train_steps:
                        break
                except tf.errors.OutOfRangeError:
                    break

        if mode == 'eval':
            cum_accu = 0.0
            nsamples = 0
            while True:
                try:
                    feed_dict = {
                        iterator.handle: iterator.get_handle(sess, 'eval'),
                        tx.context.global_mode(): tf.estimator.ModeKeys.EVAL,
                    }
                    rets = sess.run(fetches, feed_dict)
                    cum_acc += rets['accu'] * rets['batch_size']
                    nsamples += rets['batch_size']
                except tf.errors.OutOfRangeError:
                    break
            tf.logging.info('evaluation accuracy:{}'.format(cum_accu / nsamples))

        if mode == 'test':
            _all_probs = []
            while True:
                try:
                    feed_dict = {
                        iterator.handle: iterator.get_handle(sess, 'test'),
                        tx.context.global_mode(): tf.estimator.ModeKeys.PREDICT,
                    }
                    _probs = sess.run(probs, feed_dict=feed_dict)
                    _all_probs.extend(_probs.tolist())
                except:
                    break
            output_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
            with tf.gfile.GFile(output_file, "w") as writer:
                for prediction in _all_probs:
                    output_line = "\t".join(
                        str(class_probability) for class_probability in prediction) + "\n"
                    writer.write(output_line)

    with tf.Session() as sess:
        # Load Pretrained BERT model parameters
        init_checkpoint='bert_released_models/%s/bert_model.ckpt' % FLAGS.bert_pretrain_config
        if init_checkpoint:
            utils._init_bert_checkpoint(init_checkpoint)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        # Restore trained model if specified
        saver = tf.train.Saver(max_to_keep=None)
        if FLAGS.saved_model:
            saver.restore(sess, FLAGS.saved_model)

        iterator.initialize_dataset(sess)
        if FLAGS.do_train:
            iterator.restart_dataset(sess, 'train')
            _run_epoch(sess, mode='train')
            saver.save(sess, FLAGS.output_dir + '/model.ckpt')

        if FLAGS.do_eval:
            iterator.restart_dataset(sess, 'eval')
            _run_epoch(sess, mode='eval')

        if FLAGS.do_predict:
            iterator.restart_dataset(sess, 'test')
            _run_epoch(sess, mode='test')

if __name__ == "__main__":
    tf.app.run()
