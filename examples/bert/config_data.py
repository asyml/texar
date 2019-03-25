max_seq_length = 128
train_batch_size = 32
max_train_epoch = 3
display_steps = 50
eval_steps = -1

warmup_proportion = 0.1
eval_batch_size = 8

test_batch_size = 8

feature_original_types = {
    # Read feature data from TFRecords data.
    # E.g. Read feature "input_ids" as A tf.int64,
    # "FixedLenFeature" indicates its length is fixed
    # since it is not a list, and the sequence length
    # is limited by `max_seq_length`.

    "input_ids": ["tf.int64", "FixedLenFeature", max_seq_length],
    "input_mask": ["tf.int64", "FixedLenFeature", max_seq_length],
    "segment_ids": ["tf.int64", "FixedLenFeature", max_seq_length],
    "label_ids": ["tf.int64", "FixedLenFeature"]
}

feature_convert_types = {
    # Convert feature Dtype.
    # E.g. Convert data of feature "input_ids" to tf.int32
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
        "files": "data/tfrecords_files/train.tf_record"
    },

    # Repeating
    "num_epochs": -1,

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
        "files": "data/tfrecords_files/eval.tf_record"
    },

    # Non-repeating
    "num_epochs": 1,

    "shuffle": False
}

test_hparam = {
    "allow_smaller_final_batch": True,
    "batch_size": test_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_convert_types": feature_convert_types,
        "feature_original_types": feature_original_types,
        "files": "data/tfrecords_files/predict.tf_record"
    },

    # Non-repeating
    "num_epochs": 1,
    
    "shuffle": False
}


num_classes = 2
num_train_data = 3668