#!/bin/sh

train_src="../../data/giga/train.article"
train_tgt="../../data/giga/train.title"

python vocab.py \
	--src_vocab_size 30424 \
	--tgt_vocab_size 23738 \
	--train_src ${train_src} \
	--train_tgt ${train_tgt} \
	--include_singleton \
	--output giga_vocab.bin 

python process_samples.py \
    --mode sample_ngram \
    --vocab giga_vocab.bin \
    --src ${train_src} \
    --tgt ${train_tgt} \
	--sample_size 10 \
	--reward rouge \
    --output samples_giga.txt
