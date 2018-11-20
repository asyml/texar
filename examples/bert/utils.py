from munch import Munch
import json
import collections
import re
import tensorflow as tf
from texar.core.optimization import AdamWeightDecayOptimizer

def transform_bert_to_texar_config(input_json):
    config_ckpt = json.loads(
        open(input_json).read())
    configs = {}
    configs['random_seed'] = 1234
    configs['hidden_size'] = config_ckpt['hidden_size']
    hidden_dim = config_ckpt['hidden_size']
    configs['emb'] = {
        'name': 'word_embeddings',
        'dim': hidden_dim,
            }
    configs['vocab_size'] = config_ckpt['vocab_size']
    configs['token_embed'] = {
        'name': 'token_type_embeddings',
        'dim': hidden_dim,
    }
    configs['type_vocab_size'] = config_ckpt['type_vocab_size']
    configs['encoder'] = {
        'name': 'encoder',
        'dim': hidden_dim,
        'position_embedder_type': 'variables',
        'position_size': config_ckpt['max_position_embeddings'],
        'position_embedder_hparams': {
            'dim': hidden_dim,
        },
        'embed_scale': False,
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
    return Munch(configs)

def get_lr(global_step, num_train_steps, num_warmup_steps, opt_config):
    learning_rate = tf.constant(value=opt_config['learning_rate'], shape=[], dtype=tf.float32)
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
        warmup_learning_rate = opt_config['learning_rate']* warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
    return learning_rate

def get_train_op(loss, learning_rate):
    """Creates an optimizer training op."""
    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    global_step = tf.train.get_or_create_global_step()
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
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
        'bert/embeddings/word_embeddings': 'bert/embeddings/word_embeddings/word_embeddings',
        'bert/embeddings/token_type_embeddings': 'bert/embeddings/token_type_embeddings/token_type_embeddings',
        'bert/embeddings/position_embeddings': 'bert/encoder/position_embedder/position_embedder',
        'bert/embeddings/LayerNorm/beta': 'bert/encoder/LayerNorm/beta',
        'bert/embeddings/LayerNorm/gamma': 'bert/encoder/LayerNorm/gamma',
    }
    for check_name, shape in init_vars:
        if check_name.startswith('bert'):
            if check_name.startswith('bert/embeddings'):
                continue
            model_name = re.sub('layer_\d+/output/dense',
                                lambda x: x.group(0).replace('output/dense', 'ffn/output'),
                                check_name)
            if model_name == check_name:
                model_name = re.sub('layer_\d+/output/LayerNorm',
                                    lambda x: x.group(0).replace('output/LayerNorm', 'ffn/LayerNorm'),
                                    check_name)
            if model_name == check_name:
                model_name = re.sub('layer_\d+/intermediate/dense',
                                    lambda x: x.group(0).replace('intermediate/dense', 'ffn/intermediate'),
                                    check_name)
            if model_name == check_name:
                model_name = re.sub('attention/output/dense', 'attention/self/output', check_name)
            if model_name == check_name:
                model_name = check_name.replace('attention/output/LayerNorm', 'output/LayerNorm')
            assert model_name in name_to_variable.keys(), 'model name:{} not exists!'.format(model_name)

            assignment_map[check_name] = model_name
            initialized_variable_names[model_name] = 1
            initialized_variable_names[model_name + ":0"] = 1
    return (assignment_map, initialized_variable_names)
