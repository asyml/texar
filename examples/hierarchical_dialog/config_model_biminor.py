
import tensorflow as tf

num_samples = 10  # Number of samples generated for each test data instance
beam_width = num_samples

encoder_hparams = {
    "encoder_minor_type": "BidirectionalRNNEncoder",
    "encoder_minor_hparams": {
        "rnn_cell_fw": {
            "type": "GRUCell",
            "kwargs": {
                "num_units": 300,
                "kernel_initializer": tf.initializers.random_uniform(-0.08, 0.08)
            },
            "dropout": {
                "input_keep_prob": 0.5,
            }
        },
        "rnn_cell_share_config": True
    },
    "encoder_major_type": "UnidirectionalRNNEncoder",
    "encoder_major_hparams": {
        "rnn_cell": {
            "type": "GRUCell",
            "kwargs": {
                "num_units": 600,
                "kernel_initializer": tf.initializers.random_uniform(-0.08, 0.08)
            },
            "dropout": {
                "output_keep_prob": 0.3
            }
        }
    }
}
decoder_hparams = {
    "rnn_cell": {
        "type": "GRUCell",
        "kwargs": {
            "num_units": 400,
            "kernel_initializer": tf.initializers.random_uniform(-0.08, 0.08),
        },
        "dropout": {
            "input_keep_prob": 0.3
        }
    }
}
opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.001,
        }
    },
    # (It looks gradient clip does not affect the results a lot)
    # "gradient_clip": {
    #    "type": "clip_by_global_norm",
    #    "kwargs": {"clip_norm": 5.}
    # },
}
