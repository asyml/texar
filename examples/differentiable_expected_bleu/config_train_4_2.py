max_epochs = 1000
steps_per_eval = 500
tau = 1.
infer_beam_width = 1
infer_max_decoding_length = 50

mask_patterns = [(4, 2), (1, 0)]
threshold_steps = int(1e9)
minimum_interval_steps = 10000

train_xe = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": [1e-3, 1e-5]
        }
    },
    "gradient_clip": {
        "type": "clip_by_global_norm",
        "kwargs": {
            "clip_norm": 5.
        },
    },
    "name": "XE"
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
    "name": "DEBLEU"
}
