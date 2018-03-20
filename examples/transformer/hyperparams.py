# -*- coding: utf-8 -*-
import argparse
import copy
import os
from texar.core.utils import _bucket_boundaries

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
argparser.add_argument('--batch_size', type=int, default=1)
argparser.add_argument('--eval_src', type=str, default='eval_ende_wmt_bpe32k_en.txt')
argparser.add_argument('--eval_tgt', type=str, default='eval_ende_wmt_bpe32k_de.txt')

argparser.parse_args(namespace=args)
args.vocab_dir = os.path.expanduser(args.vocab_dir)
boundaries = _bucket_boundaries(max_length=256)
bucket_batch_size = [240, 180, 180, 180, 144, 144, 144, 120, 120, 120, 90, 90, 90, 90, 80, 72, 72, 60, 60, 48, 48, 48, 40, 40, 36, 30, 30, 24, 24, 20, 20, 18, 18, 16, 15, 12, 12, 10, 10, 9, 8, 8]

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
    'bucket_boundaries': boundaries,
    'bucket_batch_size': bucket_batch_size
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
    'max_seq_length':256,
    'sinusoid': True,
    'num_blocks': 6,
    'num_heads': 8,
    'poswise_feedforward': {
    'name':'ffn',
    'layers':[
        {
            'type':'Conv1D',
            'kwargs': {
                'filters':hidden_dim*4,
                'kernel_size':1,
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
            'type':'Conv1D',
            'kwargs': {
                'filters':hidden_dim,
                'kernel_size':1,
                'use_bias':True,
                }
            }
        ],
    },
}
decoder_hparams = copy.deepcopy(encoder_hparams)
decoder_hparams['share_embed_and_transform'] = True

loss_hparams = {
    'label_smoothing': 0.1,
}

opt_hparams = {
    'learning_rate_schedule': 'linear_warmup_rsqrt_decay',
    'warmup_steps': 16000,
    'max_training_steps': 250000,
}
