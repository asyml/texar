max_epochs = 1000
steps_per_eval = 500
tau = 1.
infer_beam_width = 1
infer_max_decoding_length = 50

mask_patterns = [(2, 2), (4, 2), (8, 2), (1, 0)]
threshold_steps = 10000
wait_steps = 10000

train_xe = {
    "optimizer": {
        "type": "AdamOptimizer",
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
    },
    "gradient_clip": {
        "type": "clip_by_global_norm",
        "kwargs": {
            "clip_norm": 5.
        },
    },
}
