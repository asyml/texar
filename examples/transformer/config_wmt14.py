batch_size = 3072
test_batch_size = 64

max_train_epoch = 10
display_steps = 500
eval_steps = 2000

max_decoding_length= 256

filename_prefix = "processed."
input_dir = 'temp/run_en_de_bpe/data'
vocab_file = input_dir + '/processed.vocab.pickle'
