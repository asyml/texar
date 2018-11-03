max_epochs = 1000
steps_per_eval = 500
tau = 1.
infer_beam_width = 1
infer_max_decoding_length = 50

threshold_steps = 10000
minimum_interval_steps = 10000
phases = [
    # (config_data, config_train, mask_pattern)
    ("train_0", "xe_0", None),
    ("train_0", "xe_1", None),
    ("train_0", "debleu_0", (2, 2)),
    ("train_1", "debleu_0", (4, 2)),
    ("train_1", "debleu_1", (1, 0)),
]

train_xe_0 = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 1e-3
        }
    },
    "gradient_clip": {
        "type": "clip_by_global_norm",
        "kwargs": {
            "clip_norm": 5.
        }
    },
    "name": "XE_0"
}

train_xe_1 = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 1e-5
        }
    },
    "gradient_clip": {
        "type": "clip_by_global_norm",
        "kwargs": {
            "clip_norm": 5.
        }
    },
    "name": "XE_1"
}

train_debleu_0 = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 1e-5
        }
    },
    "gradient_clip": {
        "type": "clip_by_global_norm",
        "kwargs": {
            "clip_norm": 5.
        }
    },
    "name": "DEBLEU_0"
}

train_debleu_1 = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 1e-6
        }
    },
    "gradient_clip": {
        "type": "clip_by_global_norm",
        "kwargs": {
            "clip_norm": 5.
        }
    },
    "name": "DEBLEU_1"
}
