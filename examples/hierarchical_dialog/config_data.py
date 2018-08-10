import os

data_root = './data'
max_utterance_cnt = 9

data_hparams = {
    stage: {
        "num_epochs": 1,
        "shuffle": stage != 'test',
        "batch_size": 30,
        "datasets": [
            { # source
                "variable_utterance": True,
                "max_utterance_cnt": max_utterance_cnt,
                "files": [
                    os.path.join(data_root, 
                                 '{}-source.txt'.format(stage))],
                "vocab_file": os.path.join(data_root, 'vocab.txt'),
                "embedding_init": {
                    "file": os.path.join(data_root, 'embedding.txt'),
                    "dim": 200,
                    "read_fn": "load_glove"
                },
                "data_name": "source"
            },
            { # target
                "files": [
                    os.path.join(data_root, '{}-target.txt'.format(stage))],
                "vocab_share_with": 0,
                "data_name": "target"
            },
        ] + [{ # source speaker token
                "files": os.path.join(data_root, 
                                      '{}-source-spk-{}.txt'.format(stage, i)),
                "data_type": "float",
                "data_name": "spk_{}".format(i)
            } for i in range(max_utterance_cnt)
        ] + [{ # target speaker token
                "files": os.path.join(data_root, 
                                      '{}-target-spk.txt'.format(stage)),
                "data_type": "float",
                "data_name": "spk_tgt"
            }
        ] + [{ # target refs for BLEU evaluation
                "variable_utterance": True,
                "max_utterance_cnt": 10,
                "files": [os.path.join(data_root, 
                                       '{}-target-refs.txt'.format(stage))],
                "vocab_share_with": 0,
                "data_name": "refs"
            }]
    }
    for stage in ['train', 'val', 'test']
}
