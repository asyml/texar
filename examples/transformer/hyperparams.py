# -*- coding: utf-8 -*-
import argparse
import copy
import os
from texar.utils.data_reader import _batching_scheme

class Hyperparams:
    '''Hyperparameters'''
    #source_test = 'data/translation/de-en/IWSLT16.TED.tst2014.de-en.de.xml'
    #target_test = 'data/translation/de-en/IWSLT16.TED.tst2014.de-en.en.xml'
    maxlen = 256 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    #min_cnt = 0 # words whose occurred less than min_cnt are encoded as <UNK>.

args = Hyperparams()
argparser = argparse.ArgumentParser()
argparser.add_argument('--train_src', type=str, default='train_ende_wmt_bpe32k_en.txt.filtered')
argparser.add_argument('--train_tgt', type=str, default='train_ende_wmt_bpe32k_de.txt.filtered')
#argparser.add_argument('--source_train', type=str, default='data/translation/de-en/train.tags.de-en.en')
#argparser.add_argument('--target_train', type=str, default='data/translation/de-en/train.tags.de-en.de')
argparser.add_argument('--source_test', type=str, default='/tmp/t2t_datagen/newstest2014.tok.bpe.32000.en')
argparser.add_argument('--target_test', type=str, default='/tmp/t2t_datagen/newstest2014.tok.bpe.32000.de')
argparser.add_argument('--data_dir', type=str, default='/home/hzt/shr/t2t_data/')
argparser.add_argument('--t2t_vocab', type=str, default='vocab.bpe.32000')
#batch size is only used when testing the model
argparser.add_argument('--batch_size', type=int, default=2048)
argparser.add_argument('--batch_mode', type=str, default='token')
argparser.add_argument('--eval_src', type=str, default='eval_ende_wmt_bpe32k_en.txt')
argparser.add_argument('--eval_tgt', type=str, default='eval_ende_wmt_bpe32k_de.txt')
argparser.add_argument('--max_length_bucket', type=int, default=256)
argparser.add_argument('--min_length_bucket', type=int, default=8)
argparser.add_argument('--length_bucket_step', type=float, default=1.1)
argparser.add_argument('--max_training_steps', type=int, default=250000)
argparser.add_argument('--warmup_steps', type=int, default=16000)
argparser.add_argument('--lr_constant', type=float, default=2)
argparser.parse_args(namespace=args)
args.vocab_dir = os.path.expanduser(args.data_dir)
log_params_dir = 'log_dir/bsize{}.step{}.lr_c{}warm{}/'.format(args.batch_size, args.max_training_steps,\
    args.lr_constant, args.warmup_steps)
args.log_dir = os.path.join('/space/shr/transformer_log', log_params_dir)
#args.log_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], log_params_dir)
print('args.log_dir:{}'.format(args.log_dir))
batching_scheme = _batching_scheme(
    args.batch_size,
    args.max_length_bucket,
    args.min_length_bucket,
    args.length_bucket_step,
    drop_long_sequences=True,
)
batching_scheme['boundaries'] = [b + 1 for b in batching_scheme['boundaries']]
#since there is no <BOS> token in t2t, but there is BOS in our model
print('boundaries:{}'.format(batching_scheme['boundaries']))
print('batch_sizes:{}'.format(batching_scheme['batch_sizes']))
#bucket_batch_size = [240, 180, 180, 180, 144, 144, 144, 120, 120, 120, 90, 90, 90, 90, 80, 72, 72, 60, 60, 48, 48, 48, 40, 40, 36, 30, 30, 24, 24, 20, 20, 18, 18, 16, 15, 12, 12, 10, 10, 9, 8, 8]

train_dataset_hparams = {
    "num_epochs": 100,
    "seed": 123,
    "shuffle": True,
    "source_dataset": {
        "files": [os.path.join(args.data_dir, args.train_src)],
        "vocab_file": os.path.join(args.data_dir, 'vocab.bpe.32000.filtered'),
        "processing": {
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
         }
    },
    "target_dataset": {
        "files": [os.path.join(args.data_dir, args.train_tgt)],
        "vocab_share":True,
    },
    'bucket_boundaries': batching_scheme['boundaries'],
    'bucket_batch_size': batching_scheme['batch_sizes'],
}
hidden_dim = 512
encoder_hparams = {
    'multiply_embedding_mode': "sqrt_depth",
    'embedding': {
        'name': 'lookup_table',
        'dim': hidden_dim,
        'initializer': {
            'type': 'uniform_unit_scaling',
            }
        },
    #'max_seq_length':256,
    'sinusoid': True,
    # no need to set max_seq_length in encoder when sinusoid is used
    'num_blocks': 6,
    'num_heads': 8,
    'poswise_feedforward': {
        'name':'ffn',
        'layers':[
            {
                'type':'Dense',
                'kwargs': {
                    'name':'conv1',
                    'units':hidden_dim*4,
                    'activation':'relu',
                    'use_bias':True,
                }
            },
            {
                'type':'Dropout',
                'kwargs': {
                    'rate': 0.1,
                }
            },
            {
                'type':'Dense',
                'kwargs': {
                    'name':'conv2',
                    'units':hidden_dim,
                    'use_bias':True,
                    }
            }
        ],
    },
}
decoder_hparams = copy.deepcopy(encoder_hparams)
decoder_hparams['share_embed_and_transform'] = True

loss_hparams = {
    'label_confidence': 0.9,
}

opt_hparams = {
    'learning_rate_schedule': 'constant.linear_warmup.rsqrt_decay.rsqrt_depth',
    'lr_constant': args.lr_constant,
    'warmup_steps': args.warmup_steps,
    'max_training_steps': args.max_training_steps,
    'Adam_beta1':0.9,
    'Adam_beta2':0.997,
    'Adam_epsilon':1e-9,
}
print('logdir:{}'.format(args.log_dir))
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
