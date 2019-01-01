"""
Model utility functions
"""
import json
import collections
import re
import random
import tensorflow as tf
import numpy as np
from texar import HParams

"""
Load the Json config file and transform it into Texar style configuration.
"""
def transform_bert_to_texar_config(input_json):
    config_ckpt = json.loads(
        open(input_json).read())
    configs = {}
    configs['random_seed'] = 123
    configs['hidden_size'] = config_ckpt['hidden_size']
    hidden_dim = config_ckpt['hidden_size']
    configs['embed'] = {
        'name': 'word_embeddings',
        'dim': hidden_dim}
    configs['vocab_size'] = config_ckpt['vocab_size']
    configs['segment_embed'] = {
        'name': 'token_type_embeddings',
        'dim': hidden_dim}
    configs['type_vocab_size'] = config_ckpt['type_vocab_size']

    configs['encoder'] = {
        'name': 'encoder',
        'position_embedder_type': 'variables',
        'position_size': config_ckpt['max_position_embeddings'],
        'position_embedder_hparams': {
            'dim': hidden_dim,
        },
        'embedding_dropout': config_ckpt['hidden_dropout_prob'],
        'num_blocks': config_ckpt['num_hidden_layers'],
        'multihead_attention': {
            'use_bias': True,
            'num_units': hidden_dim,
            'num_heads': config_ckpt['num_attention_heads'],
            'output_dim': hidden_dim,
            'dropout_rate': config_ckpt['attention_probs_dropout_prob'],
            'name': 'self'
        },
        'residual_dropout': config_ckpt['hidden_dropout_prob'],
        'dim': hidden_dim,
        'use_bert_config': True,
        'poswise_feedforward': {
            "layers": [
                {
                    'type': 'Dense',
                    'kwargs': {
                        'name': 'intermediate',
                        'units': config_ckpt['intermediate_size'],
                        'activation': config_ckpt['hidden_act'],
                        'use_bias': True,
                    }
                },
                {
                    'type': 'Dense',
                    'kwargs': {
                        'name': 'output',
                        'units': hidden_dim,
                        'activation': None,
                        'use_bias': True,
                    }
                },
            ],
        },
    }
    return HParams(configs, default_hparams=None)

def get_lr(global_step, num_train_steps, num_warmup_steps, static_lr):
    """
    Calculate the learinng rate given global step and warmup steps.
    The learinng rate is following a linear warmup and linear decay.
    """
    learning_rate = tf.constant(value=static_lr,
                                shape=[], dtype=tf.float32)

    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = static_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = ((1.0 - is_warmup) * learning_rate\
            +is_warmup * warmup_learning_rate)

    return learning_rate

def _get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """
    Compute the union of the current variables and checkpoint variables.
    Because the variable scope of the original BERT and Texar implementation,
    we need to build a assignment map to match the variables.
    """
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var
    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = {
        'bert/embeddings/word_embeddings': 'bert/word_embeddings/w',
        'bert/embeddings/token_type_embeddings': 'bert/token_type_embeddings/w',
        'bert/embeddings/position_embeddings':
            'bert/encoder/position_embedder/w',
        'bert/embeddings/LayerNorm/beta': 'bert/encoder/LayerNorm/beta',
        'bert/embeddings/LayerNorm/gamma': 'bert/encoder/LayerNorm/gamma',
    }
    for check_name, model_name in assignment_map.items():
        initialized_variable_names[model_name] = 1
        initialized_variable_names[model_name + ":0"] = 1

    for check_name, shape in init_vars:
        if check_name.startswith('bert'):
            if check_name.startswith('bert/embeddings'):
                continue
            model_name = re.sub(
                'layer_\d+/output/dense',
                lambda x: x.group(0).replace('output/dense', 'ffn/output'),
                check_name)
            if model_name == check_name:
                model_name = re.sub(
                    'layer_\d+/output/LayerNorm',
                    lambda x: x.group(0).replace('output/LayerNorm',
                                                 'ffn/LayerNorm'),
                    check_name)
            if model_name == check_name:
                model_name = re.sub(
                    'layer_\d+/intermediate/dense',
                    lambda x: x.group(0).replace('intermediate/dense',
                                                 'ffn/intermediate'),
                    check_name)
            if model_name == check_name:
                model_name = re.sub('attention/output/dense',
                                    'attention/self/output', check_name)
            if model_name == check_name:
                model_name = check_name.replace('attention/output/LayerNorm',
                                                'output/LayerNorm')
            assert model_name in name_to_variable.keys(),\
                'model name:{} not exists!'.format(model_name)

            assignment_map[check_name] = model_name
            initialized_variable_names[model_name] = 1
            initialized_variable_names[model_name + ":0"] = 1

    return (assignment_map, initialized_variable_names)

def init_bert_checkpoint(init_checkpoint):
    tvars = tf.trainable_variables()
    initialized_variable_names = []
    if init_checkpoint:
        (assignment_map, initialized_variable_names
        ) = _get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

def set_random_seed(myseed):
    tf.set_random_seed(myseed)
    np.random.seed(myseed)
    random.seed(myseed)
