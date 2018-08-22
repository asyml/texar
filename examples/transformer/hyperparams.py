# -*- coding: utf-8 -*-
"""
configurate the hyperparameters, based on command line arguments.
"""
import argparse
import copy
import os

def load_hyperparams():
    """
        main function to define hyperparams
    """
    # pylint: disable=too-many-statements
    class Hyperparams(): pass
    args = Hyperparams()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--wbatchsize', type=int, default=3000)
    argparser.add_argument('--epoch', type=int, default=40)
    argparser.add_argument('--start_epoch', type=int, default=0)
    argparser.add_argument('--mode', type=str,
                           default='train_and_evaluate',
                           help='can also be test mode')
    argparser.add_argument('--src_language', type=str, default='en')
    argparser.add_argument('--tgt_language', type=str, default='de')
    argparser.add_argument('--filename_prefix', type=str, default='processed.')
    argparser.add_argument('--model_dir', type=str, default='default')
    argparser.add_argument('--model_filename', type=str, default='')
    argparser.add_argument('--verbose', type=int, default=0)
    argparser.add_argument('--test_batch_size', type=int, default=32)
    argparser.add_argument('--eval_steps', type=int, default=2000)
    argparser.add_argument('--max_training_steps', type=int, default=250000)
    argparser.add_argument('--warmup_steps', type=int, default=16000)
    argparser.add_argument('--lr_constant', type=float, default=2)
    argparser.add_argument('--max_train_epoch', type=int, default=70)
    argparser.add_argument('--random_seed', type=int, default=1234)
    argparser.add_argument('--log_disk_dir', type=str)
    argparser.add_argument('--beam_width', type=int, default=5)
    argparser.add_argument('--alpha', type=float, default=0.6,\
        help=' length_penalty=(5+len(decode)/6) ^ -\alpha')
    argparser.add_argument('--save_eval_output', default=1, \
        help='save the eval output to file')
    argparser.add_argument('--eval_interval_epoch', type=int, default=1)
    argparser.add_argument('--affine_bias', type=int, default=1)
    argparser.add_argument('--eval_criteria', type=str, default='bleu')
    argparser.add_argument('--pre_encoding', type=str, default='spm')
    argparser.add_argument('--max_decode_len', type=int, default=256)
    argparser.parse_args(namespace=args)
    args.input = 'temp/run_{}_{}_{}/data'.format(
        args.src_language, args.tgt_language, args.pre_encoding)

    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=no-member
    args.filename_suffix = '.' + args.pre_encoding +'.txt'
    args.train_src = os.path.join(args.input, \
        '{}train.{}{}'.format(args.filename_prefix,
                              args.src_language, args.filename_suffix))
    args.train_tgt = os.path.join(args.input, \
        '{}train.{}{}'.format(args.filename_prefix,
                              args.tgt_language, args.filename_suffix))
    args.dev_src = os.path.join(args.input, \
        '{}dev.{}{}'.format(args.filename_prefix,
                            args.src_language, args.filename_suffix))
    args.dev_tgt = os.path.join(args.input, \
        '{}dev.{}{}'.format(args.filename_prefix,
                            args.tgt_language, args.filename_suffix))
    args.test_src = os.path.join(args.input, \
        '{}test.{}{}'.format(args.filename_prefix,
                             args.src_language, args.filename_suffix))
    args.test_tgt = os.path.join(args.input, \
        '{}test.{}{}'.format(args.filename_prefix,
                             args.tgt_language, args.filename_suffix))

    args.vocab_file = os.path.join(args.input, \
        args.filename_prefix + 'vocab.pickle')
    log_params_dir = 'log_dir/{}_{}.bsize{}.epoch{}.lr_c{}warm{}/'.format(\
        args.src_language, args.tgt_language, args.wbatchsize, \
        args.max_train_epoch, args.lr_constant, args.warmup_steps)
    args.log_dir = os.path.join(args.log_disk_dir, log_params_dir)
    args.hidden_dim = 512
    args.word_embedding_hparams = {
        'name': 'lookup_table',
        'dim': args.hidden_dim,
        'initializer': {
            'type': 'random_normal_initializer',
            'kwargs': {
                'mean':0.0,
                'stddev':args.hidden_dim**-0.5,
            },
        }
    }
    encoder_hparams = {
        'multiply_embedding_mode': "sqrt_depth",
        'embedding_dropout': 0.1,
        'position_embedder': {
            'name': 'sinusoids',
            'hparams': None,
        },
        'attention_dropout': 0.1,
        'residual_dropout': 0.1,
        'num_blocks': 6,
        'num_heads': 8,
        'num_units': args.hidden_dim,
        'initializer': {
            'type': 'variance_scaling_initializer',
            'kwargs': {
                'scale':1.0,
                'mode':'fan_avg',
                'distribution':'uniform',
            },
        },
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
                        #TODO(haoran): this dropout may be tuned
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
    decoder_hparams['transform_with_bias'] = args.affine_bias
    decoder_hparams['maximum_decode_length'] = args.max_decode_len
    decoder_hparams['beam_width'] = args.beam_width
    decoder_hparams['sampling_method'] = 'argmax'
    loss_hparams = {
        'label_confidence': 0.9,
    }

    opt_hparams = {
        'learning_rate_schedule':
            'constant.linear_warmup.rsqrt_decay.rsqrt_depth',
        'lr_constant': args.lr_constant * (args.hidden_dim ** -0.5),
        'static_lr': 1e-3,
        'warmup_steps': args.warmup_steps,
        'max_training_steps': args.max_training_steps,
        'Adam_beta1':0.9,
        'Adam_beta2':0.997,
        'Adam_epsilon':1e-9,
    }

    print('logdir:{}'.format(args.log_dir))
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    return {
        'encoder_hparams': encoder_hparams,
        'decoder_hparams': decoder_hparams,
        'loss_hparams': loss_hparams,
        'opt_hparams': opt_hparams,
        'args': args,
        }
