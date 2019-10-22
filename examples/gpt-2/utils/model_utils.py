"""
Model utility functions
"""
import sys
import json
import tensorflow as tf
import numpy as np
from texar.tf import HParams


def transform_gpt2_to_texar_config(input_json_path):
    """
    Remap the config file
    """
    config_gpt = json.loads(open(input_json_path).read())
    configs = dict()
    configs["vocab_size"] = config_gpt["n_vocab"]
    configs["context_size"] = config_gpt["n_ctx"]
    configs["embedding_size"] = config_gpt["n_embd"]
    hidden_dim = config_gpt["n_embd"]
    configs["embed"] = {
        "dim": hidden_dim,
    }
    configs["position_size"] = config_gpt["n_ctx"]
    configs["pos_embed"] = {
        "dim": hidden_dim
    }
    configs["decoder"] = {
        "dim": hidden_dim,
        "num_blocks": config_gpt["n_layer"],
        "multihead_attention": {
            "use_bias": True,
            "num_units": hidden_dim,
            "num_heads": config_gpt["n_head"],
            "output_dim": hidden_dim,
        },
        "initializer": {
            "type": "variance_scaling_initializer",
            "kwargs": {
                "scale": 1.0,
                "mode": "fan_avg",
                "distribution": "uniform",
            },
        },
        "poswise_feedforward": {
            "layers": [
                {
                    "type": "Dense",
                    "kwargs": {
                        "name": "conv1",
                        "units": hidden_dim * 4,
                        "activation": "gelu",
                        "use_bias": True,
                    }
                },
                {
                    "type": "Dense",
                    "kwargs": {
                        "name": "conv2",
                        "units": hidden_dim,
                        "use_bias": True,
                    }
                }
            ],
            "name": "ffn",
        },
    }
    return HParams(configs, default_hparams=None)


def _map_tensor_names(original_tensor_name):
    """
    Tensor name mapping
    """
    global_tensor_map = {
        "model/wte": "word_embedder/w",
        "model/wpe": "position_embedder/w",
        "model/ln_f/b": "transformer_decoder/beta",
        "model/ln_f/g": "transformer_decoder/gamma",
    }
    if original_tensor_name in global_tensor_map:
        return global_tensor_map[original_tensor_name]
    original_tensor_name_split = original_tensor_name.split("/")
    layer_tensor_map = {
        "ln_1/b": "beta",
        "ln_1/g": "gamma",
        "ln_2/b": "past_poswise_ln/beta",
        "ln_2/g": "past_poswise_ln/gamma",
        "mlp/c_fc/b": "ffn/conv1/bias",
        "mlp/c_fc/w": "ffn/conv1/kernel",
        "mlp/c_proj/b": "ffn/conv2/bias",
        "mlp/c_proj/w": "ffn/conv2/kernel",
        "attn/c_proj/b": "self_attention/multihead_attention/output/bias",
        "attn/c_proj/w": "self_attention/multihead_attention/output/kernel",
    }
    layer_num = int(original_tensor_name_split[1][1:])
    layer_feature = "/".join(original_tensor_name.split("/")[2:])
    # pylint: disable=no-else-return
    if layer_feature in layer_tensor_map:
        layer_feature_ = layer_tensor_map[layer_feature]
        tensor_name_ = "/".join(
            [
                "transformer_decoder",
                "layer_{}".format(layer_num),
                layer_feature_
            ])
        return tensor_name_
    else:
        return original_tensor_name


# pylint: disable=too-many-locals
def _get_assignment_map_from_checkpoint(sess, all_variables, init_checkpoint):
    """
    Load pretrained parameters to texar model
    """

    assignment_map = {}

    reader = tf.train.NewCheckpointReader(init_checkpoint)
    var_names_list = reader.get_variable_to_shape_map().keys()
    ckpt_names_vs_vals = {}
    for var_name in var_names_list:
        ckpt_names_vs_vals[var_name] = reader.get_tensor(var_name)

    def _assign_by_name(sess, tensor_name, data):
        local_tensor = [var for var in all_variables
                        if tensor_name in var.name][0]
        sess.run(tf.assign(local_tensor, data))

    def _get_tensor_by_name(tensor_name):
        local_tensor = [var for var in all_variables
                        if tensor_name in var.name][0]
        return local_tensor

    for idx, ckpt_tensor_name in enumerate(ckpt_names_vs_vals):
        processing = (idx + 1.0) / len(ckpt_names_vs_vals.keys())
        sys.stdout.write("\rLoading checkpoint: {:.1%}".format(processing))
        sys.stdout.flush()

        ckpt_tensor_name_feature = ""
        if len(ckpt_tensor_name.split("/")) > 2:
            ckpt_tensor_name_feature = "/".join(
                ckpt_tensor_name.split("/")[2:])
        if ckpt_tensor_name_feature == "attn/c_attn/w":
            layer_num = int(ckpt_tensor_name.split("/")[1][1:])
            template = ("transformer_decoder/layer_{}/self_attention/"
                        "multihead_attention/{}/kernel")
            local_tensor_name_q_w = template.format(layer_num, "query")
            local_tensor_name_k_w = template.format(layer_num, "key")
            local_tensor_name_v_w = template.format(layer_num, "value")

            data = ckpt_names_vs_vals[ckpt_tensor_name]
            assert data.shape[2] % 3 == 0, ("tensor 'attn/c_attn/w' "
                                            "shape is not dividable")
            index_w = data.shape[2] // 3
            q_w = data[:, :, :index_w]
            k_w = data[:, :, index_w: 2 * index_w]
            v_w = data[:, :, 2 * index_w:]
            _assign_by_name(sess, local_tensor_name_q_w, np.squeeze(q_w))
            _assign_by_name(sess, local_tensor_name_k_w, np.squeeze(k_w))
            _assign_by_name(sess, local_tensor_name_v_w, np.squeeze(v_w))

        elif ckpt_tensor_name_feature == "attn/c_attn/b":
            layer_num = int(ckpt_tensor_name.split("/")[1][1:])
            template = ("transformer_decoder/layer_{}/self_attention/"
                        "multihead_attention/{}/bias")
            local_tensor_name_q_b = template.format(layer_num, "query")
            local_tensor_name_k_b = template.format(layer_num, "key")
            local_tensor_name_v_b = template.format(layer_num, "value")

            data = ckpt_names_vs_vals[ckpt_tensor_name]
            assert data.shape[0] % 3 == 0, ("tensor 'attn/c_attn/b'"
                                            " shape is not dividable")
            index_b = data.shape[0] // 3
            q_b = data[:index_b]
            k_b = data[index_b: 2 * index_b]
            v_b = data[2 * index_b:]
            _assign_by_name(sess, local_tensor_name_q_b, q_b)
            _assign_by_name(sess, local_tensor_name_k_b, k_b)
            _assign_by_name(sess, local_tensor_name_v_b, v_b)

        else:
            local_tensor_name = _map_tensor_names(ckpt_tensor_name)
            local_tensor = _get_tensor_by_name(local_tensor_name)
            assignment_map[ckpt_tensor_name] = local_tensor

    return assignment_map


def init_gpt2_checkpoint(sess, init_checkpoint):
    """
    Initializes GPT-2 model parameters from a checkpoint

    Args:
        init_checkpoint (str): Path to the checkpoint.
    """
    tvars = tf.trainable_variables()
    if init_checkpoint:
        assignment_map = _get_assignment_map_from_checkpoint(
            sess,
            tvars,
            init_checkpoint)
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
            init_checkpoint, assignment_map, reshape_variables=True)
        init_fn(sess)
