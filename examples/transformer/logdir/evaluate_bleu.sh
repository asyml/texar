YEAR=2014
BEAM_SIZE=1
ALPHA=0.6

PROBLEM=translate_ende_wmt_bpe32k

MODEL=transformer
HPARAMS=transformer_base_single_gpu
OUTPUT_FILE=my-model-152000.output.txt
#OUTPUT_FILE=my-model-12

perl ~/mosesdecoder/scripts/tokenizer/tokenizer.perl -l de < /tmp/t2t_datagen/newstest${YEAR}.de > /tmp/t2t_datagen/newstest${YEAR}.de.tok
#Do compound splitting on the reference
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < /tmp/t2t_datagen/newstest${YEAR}.de.tok > /tmp/t2t_datagen/newstest${YEAR}.de.atat

#process the output file
cat ${OUTPUT_FILE} | sed 's/@@ //g' > outputs.words
# compound splitting
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < outputs.words > outputs.atat

#评分
perl ~/mosesdecoder/scripts/generic/multi-bleu.perl /tmp/t2t_datagen/newstest${YEAR}.de.atat < outputs.atat

