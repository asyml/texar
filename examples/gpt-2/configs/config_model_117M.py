"""Texar config file of the GPT-2 model_117M model.
"""

vocab_size = 50257
dim = 768

embed = {
    "dim": dim,
}

pos_embed = {
    "dim": dim
}
position_size = 1024

decoder = {
    "dim": dim,
    "num_blocks": 12,
    "multihead_attention": {
        "use_bias": True,
        "num_units": dim,
        "num_heads": 12,
        "output_dim": dim,
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
                    "units": dim * 4,
                    "activation": "gelu",
                    "use_bias": True,
                }
            },
            {
                "type": "Dense",
                "kwargs": {
                    "name": "conv2",
                    "units": dim,
                    "use_bias": True,
                }
            }
        ],
        "name": "ffn",
    },
}
