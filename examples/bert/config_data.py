tfrecord_data_dir = "data/MRPC"
max_seq_length = 128
num_classes = 2
num_train_data = 3668

train_batch_size = 32
max_train_epoch = 3
display_steps = 50  # Print training loss every display_steps; -1 to disable
eval_steps = -1    # Eval on the dev set every eval_steps; -1 to disable
# Proportion of training to perform linear learning
# rate warmup for. E.g., 0.1 = 10% of training.
warmup_proportion = 0.1

eval_batch_size = 8
test_batch_size = 8


feature_original_types = {
    # Reading features from TFRecord data file.
    # E.g., Reading feature "input_ids" as dtype `tf.int64`;
    # "FixedLenFeature" indicates its length is fixed for all data instances;
    # and the sequence length is limited by `max_seq_length`.
    "input_ids": ["tf.int64", "FixedLenFeature", max_seq_length],
    "input_mask": ["tf.int64", "FixedLenFeature", max_seq_length],
    "segment_ids": ["tf.int64", "FixedLenFeature", max_seq_length],
    "label_ids": ["tf.int64", "FixedLenFeature"]
}

feature_convert_types = {
    # Converting feature dtype after reading. E.g.,
    # Converting the dtype of feature "input_ids" from `tf.int64` (as above)
    # to `tf.int32`
    "input_ids": "tf.int32",
    "input_mask": "tf.int32",
    "label_ids": "tf.int32",
    "segment_ids": "tf.int32"
}

train_hparam = {
    "allow_smaller_final_batch": False,
    "batch_size": train_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_convert_types": feature_convert_types,
        "feature_original_types": feature_original_types,
        "files": "{}/train.tf_record".format(tfrecord_data_dir)
    },
    "shuffle": True,
    "shuffle_buffer_size": 100
}

eval_hparam = {
    "allow_smaller_final_batch": True,
    "batch_size": eval_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_convert_types": feature_convert_types,
        "feature_original_types": feature_original_types,
        "files": "{}/eval.tf_record".format(tfrecord_data_dir)
    },
    "shuffle": False
}

test_hparam = {
    "allow_smaller_final_batch": True,
    "batch_size": test_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_convert_types": feature_convert_types,
        "feature_original_types": feature_original_types,
        "files": "{}/predict.tf_record".format(tfrecord_data_dir)
    },

    "shuffle": False
}
