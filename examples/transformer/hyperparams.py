# -*- coding: utf-8 -*-
import argparse
import copy
import os
from texar.utils.data_reader import _batching_scheme

class Hyperparams:
    '''Hyperparameters'''
    #source_test = 'data/translation/de-en/IWSLT16.TED.tst2014.de-en.de.xml'
    #target_test = 'data/translation/de-en/IWSLT16.TED.tst2014.de-en.en.xml'
    #min_cnt = 0 # words whose occurred less than min_cnt are encoded as <UNK>.
    max_seq_length = 256

args = Hyperparams()
argparser = argparse.ArgumentParser()
#argparser.add_argument('--data_dir', type=str, default='~/others_repo/Attention_is_All_You_Need/data/en_vi/data')
argparser.add_argument('--running_mode', type=str, default='train')
argparser.add_argument('--src_language', type=str, default='en')
#argparser.add_argument('--tgt_language', type=str, default='vi')
argparser.add_argument('--tgt_language', type=str, default='de')
argparser.add_argument('--filename_prefix',type=str, default='processed.')
argparser.add_argument('--debug', type=int, default=0)
#argparser.add_argument('--train_src', type=str, default='train_ende_wmt_bpe32k_en.txt.filtered')
#argparser.add_argument('--train_tgt', type=str, default='train_ende_wmt_bpe32k_de.txt.filtered')
#argparser.add_argument('--source_test', type=str, default='/tmp/t2t_datagen/newstest2014.tok.bpe.32000.en')
#argparser.add_argument('--target_test', type=str, default='/tmp/t2t_datagen/newstest2014.tok.bpe.32000.de')

#argparser.add_argument('--data_dir', type=str, default='/home/hzt/shr/t2t_data/')
argparser.add_argument('--data_dir', type=str, default='/home/shr/t2t_data/')

#argparser.add_argument('--t2t_vocab', type=str, default='vocab.bpe.32000')
#batch size is only used when testing the model
argparser.add_argument('--batch_size', type=int, default=4096)
argparser.add_argument('--batch_mode', type=str, default='token')
argparser.add_argument('--test_batch_size', type=int, default=10)
argparser.add_argument('--max_length_bucket', type=int, default=256)
argparser.add_argument('--min_length_bucket', type=int, default=8)
argparser.add_argument('--length_bucket_step', type=float, default=1.1)
argparser.add_argument('--max_training_steps', type=int, default=250000)
argparser.add_argument('--warmup_steps', type=int, default=16000)
argparser.add_argument('--save_checkpoint_interval', type=int, default=1500)
argparser.add_argument('--lr_constant', type=float, default=2)
argparser.add_argument('--max_train_epoch', type=int, default=40)
argparser.add_argument('--num_epochs', type=int, default=1)
argparser.add_argument('--random_seed', type=int, default=123)
argparser.add_argument('--log_disk_dir', type=str)
argparser.add_argument('--beam_width', type=int, default=2)
argparser.add_argument('--alpha', type=float, default=0.6,\
    help=' length_penalty=(5+len(decode)/6) ^ -\alpha')
argparser.add_argument('--save_eval_output', default=1, \
    help='save the eval output to file')
#argparser.add_argument('--batch_relax', type=int, default=True)

argparser.parse_args(namespace=args)

print('args.data_dir:{}'.format(args.data_dir))
args.data_dir = os.path.expanduser(args.data_dir)
args.filename_prefix = args.filename_prefix + args.src_language + '_' + args.tgt_language
args.train_src = os.path.join(args.data_dir, args.filename_prefix + '.train.' + args.src_language + '.txt')
args.train_tgt = os.path.join(args.data_dir, args.filename_prefix + '.train.' + args.tgt_language + '.txt')
args.dev_src = os.path.join(args.data_dir, args.filename_prefix + '.dev.' + args.src_language + '.txt')
args.dev_tgt = os.path.join(args.data_dir, args.filename_prefix + '.dev.' + args.tgt_language + '.txt')
args.test_src = os.path.join(args.data_dir, args.filename_prefix + '.test.' + args.src_language +'.txt')
args.test_tgt = os.path.join(args.data_dir, args.filename_prefix + '.test.' + args.tgt_language + '.txt')

args.vocab_file = os.path.join(args.data_dir, args.filename_prefix + '.vocab.text')
print('vocabulary{}'.format(args.vocab_file))

log_params_dir = 'log_dir/{}_{}.bsize{}.epoch{}.lr_c{}warm{}/'.format(args.src_language, args.tgt_language, \
    args.batch_size, args.max_train_epoch, args.lr_constant, args.warmup_steps)
if args.debug:
    args.log_disk_dir += '/debug/'
args.log_dir = os.path.join(args.log_disk_dir, log_params_dir)
print('args.log_dir:{}'.format(args.log_dir))
batching_scheme = _batching_scheme(
    args.batch_size,
    args.max_length_bucket,
    args.min_length_bucket,
    args.length_bucket_step,
    drop_long_sequences=True,
    #batch_relax=args.batch_relex,
)
batching_scheme['boundaries'] = [b + 1 for b in batching_scheme['boundaries']]

train_dataset_hparams = {
    "num_epochs": args.num_epochs,
    #"num_epochs": args.max_train_epoch,
    "seed": args.random_seed,
    "shuffle": True,
    "source_dataset": {
        "files": [args.train_src],
        "vocab_file": args.vocab_file,
        #"max_seq_length": 256,
        #"length_filter_mode": "truncate",
    },
    "target_dataset": {
        "files": [args.train_tgt],
        "vocab_share":True,
        #"max_seq_length": 256,
        #"length_filter_mode": "truncate",
    },
    'bucket_boundaries': batching_scheme['boundaries'],
    'bucket_batch_sizes': batching_scheme['batch_sizes'],
    'allow_smaller_final_batch': True,
}
eval_dataset_hparams = {
    "num_epochs": 1,
    'seed': 123,
    'shuffle': False,
    'source_dataset' : {
        'files': [os.path.join(args.data_dir, args.dev_src)],
        'vocab_file': args.vocab_file,
    },
    'target_dataset': {
        'files': [os.path.join(args.data_dir, args.dev_tgt)],
        'vocab_share': True,
    },
    'batch_size': args.test_batch_size,
    'allow_smaller_final_batch': True,
}
test_dataset_hparams = {
    "num_epochs": 1,
    "seed": 123,
    "shuffle": False,
    "source_dataset": {
        "files": [os.path.join(args.data_dir, args.test_src)],
        "vocab_file": args.vocab_file,
    },
    "target_dataset": {
        "files": [os.path.join(args.data_dir, args.test_tgt)],
        "vocab_share":True,
    },
    'batch_size': args.test_batch_size,
    'allow_smaller_final_batch': True,
}
args.hidden_dim = 512
args.word_embedding_hparams={
    'name': 'lookup_table',
    'dim': args.hidden_dim,
    'initializer': {
        'type': 'uniform_unit_scaling',
        'kwargs': {},
    }
    # in get_initializer function: kwargs cannot be accessed, bug
}
encoder_hparams = {
    'multiply_embedding_mode': "sqrt_depth",
    'sinusoid': True,
    'num_blocks': 6,
    'num_heads': 8,
    'poswise_feedforward': {
        'name':'ffn',
        'layers':[
            {
                'type':'Dense',
                'kwargs': {
                    'name':'conv1',
                    'units':args.hidden_dim*4,
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
                    'units':args.hidden_dim,
                    'use_bias':True,
                    }
            }
        ],
    },
}
decoder_hparams = copy.deepcopy(encoder_hparams)
decoder_hparams['share_embed_and_transform'] = True
decoder_hparams['maximum_decode_length'] = args.max_seq_length
decoder_hparams['beam_width'] = args.beam_width

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
