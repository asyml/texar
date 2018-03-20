import tensorflow as tf
import argparse
import os
from tensor2tensor.data_generators import text_encoder
class Args():
    pass
args = Args()
argparser = argparse.ArgumentParser()
argparser.add_argument('--in_vocab', type=str, default='vocab.bpe.32000')
argparser.add_argument('--eval_src', type=str, default='eval_ende_wmt_bpe32k_en.txt')
argparser.add_argument('--eval_tgt', type=str, default='eval_ende_wmt_bpe32k_de.txt')
argparser.add_argument('--data_dir', type=str, default='~/shr/t2t_data/')
argparser.add_argument('--train_src', type=str, default='train_ende_wmt_bpe32k_en.txt.filtered')
argparser.add_argument('--train_tgt', type=str, default='train_ende_wmt_bpe32k_de.txt.filtered')

argparser.parse_args(namespace=args)
args.data_dir = os.path.expanduser(args.data_dir)
token_vocab = text_encoder.TokenTextEncoder(
    os.path.join(args.data_dir, args.in_vocab),
    replace_oov='UNK'
)

with open(os.path.join(args.data_dir, args.eval_src), 'w+') as srcfile, \
        open(os.path.join(args.data_dir, args.eval_tgt), 'w+') as tgtfile:
    filename = os.path.join(args.data_dir, \
        'translate_ende_wmt_bpe32k-dev-%05d-of-%05d' % (0,1))
    for serialized_example in tf.python_io.tf_record_iterator(filename):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        source = example.features.feature['inputs']
        target = example.features.feature['targets']
        int64list = source.int64_list.value[:-1] #EOS should not be printed
        source_string =token_vocab.decode(int64list)
        srcfile.write(source_string+'\n')
        int64list = target.int64_list.value[:-1] #EOS should not be printed
        target_string = token_vocab.decode(int64list)
        tgtfile.write(target_string+'\n')

with open(os.path.join(args.data_dir + args.train_src),'w+') as srcfile, \
    open(os.path.join(args.data_dir + args.train_tgt),'w+') as tgtfile:
    for fileidx in range(100):
        filename = os.path.join(args.data_dir, \
            'translate_ende_wmt_bpe32k-train-%05d-of-00100' % fileidx)
        for serialized_example in tf.python_io.tf_record_iterator(filename):
            example = tf.train.Example()
            example.ParseFromString(serialized_example)
            source = example.features.feature['inputs']
            target = example.features.feature['targets']
            int64list = source.int64_list.value[:-1]
            if len(int64list)>256:
                print('len:{}'.format(len(int64list)))
                continue
            source_string = token_vocab.decode(int64list)
            int64list = target.int64_list.value[:-1]
            if len(int64list)>256:
                print('len:{}'.format(len(int64list)))
                continue
            target_string = token_vocab.decode(int64list)
            srcfile.write(source_string+'\n')
            tgtfile.write(target_string+'\n')

