export CUDA_VISIBLE_DEVICES=0

PROBLEM=translate_ende_wmt_bpe32k

MODEL=transformer
HPARAMS=transformer_base_single_gpu

DATA_DIR=$HOME/shr/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/shr/t2t_train/$PROBLEM/$MODEL-$HPARAMS/b3072.step50k/
YEAR=2014
BEAM_SIZE=5
ALPHA=0.6

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM && \
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --train_steps=500000 \
  --hparams='batch_size=3072'
if [ $? -eq 0 ];then
    t2t-decoder \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=/tmp/t2t_datagen/newstest${YEAR}.tok.bpe.32000.en
else
    echo 'model is not trained successfully'
    exit 1
fi

if [ $? -eq 0 ];then
    cat /tmp/t2t_datagen/newstest${YEAR}.tok.bpe.32000.en.${MODEL}.${HPARAMS}.${PROBLEM}.beam${BEAM_SIZE}.alpha${ALPHA}.decodes | sed 's/@@ //g' > /tmp/t2t_datagen/newstest${YEAR}.tok.bpe.32000.en.${MODEL}.${HPARAMS}.${PROBLEM}.beam${BEAM_SIZE}.alpha${ALPHA}.words && \
    t2t-bleu --reference=/tmp/t2t_datagen/newstest${YEAR}.de --translation=/tmp/t2t_datagen/newstest${YEAR}.tok.bpe.32000.en.${MODEL}.${HPARAMS}.${PROBLEM}.beam${BEAM_SIZE}.alpha${ALPHA}.words
else
    echo 'cannot find model output'
    exit 1
fi
