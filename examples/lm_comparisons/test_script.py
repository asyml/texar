import texar as tx

train_file = './simple-examples/data/ptb.train.txt'

train_text = tx.data.read_words(train_file, newline_token='<EOS>')

print('lenth of train_text {}'.format(len(train_text)))


