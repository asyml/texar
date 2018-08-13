beam_width = 5

encoder_hparams = {
    "encoder_minor_type": "BidirectionalRNNEncoder",
    "encoder_minor_hparams": {
        "rnn_cell_fw": {
            "type": "GRUCell",
            "kwargs": {
                "num_units": 300
            },
            "dropout": {
                "input_keep_prob": 0.9,
            }
        },
        "rnn_cell_share_config": True
    },
    "encoder_major_type": "UnidirectionalRNNEncoder",
    "encoder_major_hparams": {
        "rnn_cell": {
            "type": "GRUCell",
            "kwargs": {
                "num_units": 400
            },
            "dropout": {
                "output_keep_prob": 0.9
            }
        }
    }
}
decoder_hparams = {
    "rnn_cell": {
        "type": "GRUCell",
        "kwargs": {
            "num_units": 400,
        },
        "dropout": {
            "input_keep_prob": 0.9, 
        }   
    }
}
opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.001,
        }
    }
}
