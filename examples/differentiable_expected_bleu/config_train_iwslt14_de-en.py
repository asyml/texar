max_epochs = 1000
steps_per_eval = 500
tau = 1.
infer_beam_width = 1
infer_max_decoding_length = 50

mask_patterns = [(2, 2), (4, 2), (8, 2), (1, 0)]
threshold_steps = 10000
minimum_interval_steps = 10000

train_xe = {
    "optimizer": {
        "type": "AdamOptimizer",
    },
    "learning_rate_decay": {
        "type": "piecewise_constant",
        "kwargs": {
            "boundaries": [10000],
            "values": [1e-3, 1e-5],
        },
    },
    "gradient_clip": {
        "type": "clip_by_global_norm",
        "kwargs": {
            "clip_norm": 5.
        },
    },
}

train_debleu = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 1e-5,
        }
    },
    "gradient_clip": {
        "type": "clip_by_global_norm",
        "kwargs": {
            "clip_norm": 5.
        },
    },
}