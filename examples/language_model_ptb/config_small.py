#
"""PTB LM small config.
"""

# pylint: disable=invalid-name, too-few-public-methods, missing-docstring

init_scale = 0.1
num_epochs = 13
keep_prob = 1.0
batch_size = 20
num_steps = 20

cell = {
    "type": "LSTMBlockCell",
    "kwargs": {
        "num_units": 200,
        "forget_bias": 0.
    },
    "dropout": {"output_keep_prob": keep_prob},
    "num_layers": 2
}
emb = {
    "dim": 200
}
opt = {
    "optimizer": {
        "type": "GradientDescentOptimizer",
        "kwargs": {"learning_rate": 1.0}
    },
    "gradient_clip": {
        "type": "clip_by_global_norm",
        "kwargs": {"clip_norm": 5.}
    },
    "learning_rate_decay": {
        "type": "exponential_decay",
        "kwargs": {
            "decay_steps": 1,
            "decay_rate": 0.5,
            "staircase": True
        },
        "start_decay_step": 3
    }
}
