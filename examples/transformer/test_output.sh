#!/usr/bin/env bash

TF=$(pwd)

export PATH=$PATH:$TF/../../tools/
encoder=wpm
src_language=en
tgt_language=vi
# update these variables
DATA=${TF}"/data/${src_language}_${tgt_language}"
NAME="run_${src_language}_${tgt_language}_${encoder}"
out="temp/${name}"

train_src=$DATA/train.${src_language}
train_tgt=$DATA/train.${tgt_language}
TEST_SRC=$DATA/tst2013.${src_language}
TEST_TGT=$DATA/tst2013.${tgt_language}
DEV_SRC=$DATA/tst2012.${src_language}
DEV_TGT=$DATA/tst2012.${tgt_language}

vocab_size=32000

mv $out/test/test.out{,.${encoder}}
mv $out/test/dev.out{,.${encoder}}
case ${encoder} in
    'spm')
        spm_decode --model ${out}/data/wpm-codes.${vocab_size}.model --input_format=piece < ${out}/test/dev.out.wpm > $out/test/dev.out
        spm_decode --model ${out}/data/wpm-codes.${vocab_size}.model --input_format=piece < ${out}/test/test.out.wpm > $out/test/test.out;;
    'bpe')
        cat  | sed -E 's/(@@ )|(@@ ?$)//g' > decodes

esac
