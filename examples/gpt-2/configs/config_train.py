"""Config file for GPT2 training.
"""
# pylint: disable=invalid-name

tfrecord_data_dir = "data/toy"
max_seq_length = 128
max_decoding_length = max_seq_length

train_batch_size = 32
max_train_epoch = 100
display_steps = 10  # Print training loss every display_steps; -1 to disable
eval_steps = -1    # Eval on the dev set every eval_steps; -1 to disable
# Checkpoint model parameters every checkpoint_steps; -1 to disable
checkpoint_steps = -1

eval_batch_size = 8
test_batch_size = 8

# Optimization configs

opt = {
    'optimizer': {
        'type': 'AdamOptimizer',
        'kwargs': {
            'learning_rate': 0.001
        }
    }
}

# Data configs

feature_original_types = {
    # Reading features from TFRecord data file.
    # E.g., Reading feature "text_ids" as dtype `tf.int64`;
    # "FixedLenFeature" indicates its length is fixed for all data instances;
    # and the sequence length is limited by `max_seq_length`.
    "text_ids": ["tf.int64", "FixedLenFeature", max_seq_length],
    "length": ["tf.int64", "FixedLenFeature"]
}
feature_convert_types = {
    # Converting feature dtype after reading. E.g.,
    # Converting the dtype of feature "text_ids" from `tf.int64` (as above)
    # to `tf.int32`
    "text_ids": "tf.int32",
    "length": "tf.int32"
}

train_hparam = {
    "allow_smaller_final_batch": False,
    "batch_size": train_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_original_types": feature_original_types,
        "feature_convert_types": feature_convert_types,
        "files": "{}/train.tf_record".format(tfrecord_data_dir)
    },
    "shuffle": True,
    "shuffle_buffer_size": 1000
}

dev_hparam = {
    "allow_smaller_final_batch": True,
    "batch_size": eval_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_original_types": feature_original_types,
        "feature_convert_types": feature_convert_types,
        "files": "{}/dev.tf_record".format(tfrecord_data_dir)
    },
    "shuffle": False
}

# Set to `test_hparam` to `None` if generating from scratch
# (instead of generating continuation) at test time
test_hparam = {
    "allow_smaller_final_batch": True,
    "batch_size": test_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_original_types": feature_original_types,
        "feature_convert_types": feature_convert_types,
        "files": "{}/test.tf_record".format(tfrecord_data_dir)
    },
    "shuffle": False
}
