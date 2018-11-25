#!/bin/sh

train_src="../../data/iwslt14/train.de"
train_tgt="../../data/iwslt14/train.en"

python vocab.py \
	--src_vocab_size 32007 \
	--tgt_vocab_size 22820 \
	--train_src ${train_src} \
	--train_tgt ${train_tgt} \
	--include_singleton \
	--output iwslt14_vocab.bin 

python process_samples.py \
    --mode sample_ngram \
    --vocab iwslt14_vocab.bin \
    --src ${train_src} \
    --tgt ${train_tgt} \
	--sample_size 10 \
	--reward bleu \
    --output samples_iwslt14.txt
