batch_size = 4096
n_gpu = 2
test_batch_size = 64

max_train_epoch = 20
display_steps = 10
eval_steps = 10

max_decoding_length = 256

filename_prefix = "processed."
input_dir = 'temp/run_en_vi_spm/data'
vocab_file = input_dir + '/processed.vocab.pickle'
