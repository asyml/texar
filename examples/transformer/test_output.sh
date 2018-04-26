#!/usr/bin/env bash

TF=$(pwd)

export PATH=$PATH:$TF/../../tools/

encoder=wpm
if [ -z $1 ]; then
    src_language=en
else
    src_language=$1
fi

if [ -z $2] ; then
    src_language=vi
else
    tgt_language=$2
fi
# update these variables

data=${TF}"/data/${src_language}_${tgt_language}/"
out="temp/run_${src_language}_${tgt_language}_${encoder}"

batch_size=2048
max_epoch=70
lr_constant=2
warmup_steps=16000

model_dir="/space/shr/transformer_${encoder}/log_dir/${src_language}_${tgt_language}.bsize${batch_size}.epoch${max_epoch}.lr_c${lr_constant}warm${warmup_steps}/"
model_filename="my-model-highest_bleu.ckpt"
beam_size=5
alpha=0.6
decodes_file=${model_filename}.test.beam${beam_size}alpha${alpha}.outputs.decodes

decodes_file_fullpath=${model_dir}/${decodes_file}
test_tgt=$data/test.${tgt_language}

vocab_size=32000
case ${encoder} in
    'wpm')
        spm_decode --model ${out}/data/wpm-codes.${vocab_size}.model --input_format=piece < ${decodes_file_fullpath} > $model_dir/test.out;;
    'bpe')
        cat  ${decodes_file_fullpath} | sed -E 's/@@ //g' > ${model_dir}/test.out;;
esac
echo "reference ${test_tgt}"
echo "translation ${model_dir}/test.out"
t2t-bleu --reference=${test_tgt} --translation=${model_dir}/test.out

