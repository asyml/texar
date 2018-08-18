#!/usr/bin/env bash

###########################################################################

# This file provides a script to preprocess raw text corpora to generate
# vocabulary with sentence piece encoding or byte pairwise encoding.
# You can provide customed source language and target language if the
# correpsonding corpora are saved in the correct path.
# By default, the vocab_size is set to 32000.
#
# Run it with `preprocess_data.sh source_language target_language`.
# For example, `preprocess_data.sh en vi`

###########################################################################


TF=$(pwd)

export PATH=$PATH:$TF/../../bin/utils/
encoder=spm
src_language=$1
tgt_language=$2

# update these variables
data=${TF}"/data/${src_language}_${tgt_language}"
name="run_${src_language}_${tgt_language}_${encoder}"
out="temp/${name}"

train_src=$data/train.${src_language}
train_tgt=$data/train.${tgt_language}
valid_src=$data/dev.${src_language}
valid_tgt=$data/dev.${tgt_language}
test_src=$data/test.${src_language}
test_tgt=$data/test.${tgt_language}
vocab_size=32000

#====== EXPERIMENT BEGIN ======
echo "Output dir = $out"
[ -d $out ] || mkdir -p $out
[ -d $out/data ] || mkdir -p $out/data
[ -d $out/test ] || mkdir -p  $out/test

echo "Step 1a: Preprocess inputs"

echo "Learning Word Piece or Byte Pairwise on source and target combined"
case ${encoder} in
    'spm')
        spm_train --input=${train_src},${train_tgt} --vocab_size ${vocab_size} --model_prefix=$out/data/spm-codes.${vocab_size}
        spm_encode --model $out/data/spm-codes.${vocab_size}.model --output_format=piece < $train_src > $out/data/train.${src_language}.spm
        spm_encode --model $out/data/spm-codes.${vocab_size}.model --output_format=piece < $valid_src > $out/data/valid.${src_language}.spm
        spm_encode --model $out/data/spm-codes.${vocab_size}.model --output_format=piece < $test_src > $out/data/test.${src_language}.spm
        spm_encode --model $out/data/spm-codes.${vocab_size}.model --output_format=piece <  $train_tgt > $out/data/train.${tgt_language}.spm
        spm_encode --model $out/data/spm-codes.${vocab_size}.model --output_format=piece <  $valid_tgt > $out/data/valid.${tgt_language}.spm
        spm_encode --model $out/data/spm-codes.${vocab_size}.model --output_format=piece < ${test_tgt} > $out/data/test.${tgt_language}.spm
        cp ${test_tgt} ${out}/test/test.${tgt_language} ;;
    'bpe'):
        cat ${train_src} ${train_tgt} | learn_bpe -s ${vocab_size} > ${out}/data/bpe-codes.${vocab_size}
        apply_bpe -c ${out}/data/bpe-codes.${vocab_size} < ${train_src} > $out/data/train.${src_language}.bpe
        apply_bpe -c ${out}/data/bpe-codes.${vocab_size} < ${valid_src} > ${out}/data/valid.${src_language}.bpe
        apply_bpe -c ${out}/data/bpe-codes.${vocab_size} < ${test_src} > ${out}/data/test.${src_language}.bpe
        apply_bpe -c ${out}/data/bpe-codes.${vocab_size} < ${train_tgt} > $out/data/train.${tgt_language}.bpe
        apply_bpe -c ${out}/data/bpe-codes.${vocab_size} < ${valid_tgt} > ${out}/data/valid.${tgt_language}.bpe
        apply_bpe -c ${out}/data/bpe-codes.${vocab_size} < ${test_tgt} > ${out}/data/test.${tgt_language}.bpe
        cp ${test_tgt} ${out}/test/test.${tgt_language} ;;
esac

python ${TF}/preprocess.py -i ${out}/data \
    --src ${src_language}.${encoder} \
    --tgt ${tgt_language}.${encoder} \
    --save_data processed. \
    --max_seq_length=70 \
    --pre_encoding=${encoder}
