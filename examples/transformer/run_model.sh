mode=$1

if [ -z $3 ]; then
    src_language=en
else
    src_language=$3
fi

if [ -z $4 ]; then
    tgt_language=vi
else
    tgt_language=$4
fi

MAX_TRAINING_STEPS=125000
MAX_EPOCH=15
BATCH_SIZE=3072
ENCODER=bpe
beam_size=5
#MAX_EPOCH=70
#BATCH_SIZE=2048
encoder=wpm
DATA_DIR="./temp/run_${src_language}_${tgt_language}_${encoder}/data/"

#LOG_DISK_DIR='/home2/shr/transformer/'
LOG_DISK_DIR=/space/shr/transformer_${ENCODER}/
hparams_set=$1
running_mode=$2

case ${hparams_set} in
    100)
    echo 'running the model according to tensor2tensor default hparams'
    echo "mode ${running_mode} ${src_language}-${tgt_language} "
    python transformer_overall.py --running_mode=${running_mode} --max_train_epoch=70 --max_training_steps=125000 \
        --pre_encoding=${encoder} \
        --data_dir=${DATA_DIR} \
        --src_language=${src_language} --tgt_language=${tgt_language} \
        --batch_size=2048 --test_batch_size=32 \
        --beam_width=5 --alpha=0.6\
        --log_disk_dir=/space/shr/transformer_${encoder}/ \
        --draw_for_debug=0 --affine_bias=0 --eval_interval_epoch=1 \
        --zero_pad=1 --bos_pad=1 \
        --filename_prefix=processed. &> logging_100_${running_mode}.txt;;
    200)
        echo 'running the model with bigger batch_size and training steps'
        echo 'only support en and de language for now'
        src_language=en
        tgt_language=de
        echo "mode ${running_mode} ${src_language}-${tgt_language}"
        python transformer_overall.py --running_mode=${running_mode} --max_train_epoch=70 --max_training_steps=125000 \
            --pre_encoding=${encoder} --data_dir=${DATA_DIR} \
            --src_language=${src_language} --tgt_language=${tgt_language} \
            --batch_size=3072 --test_batch_size=32 --max_training_steps=500000\
            --beam_width=5 --alpha=0.6 \
            --log_disk_dir=/space/shr/transformer_${encoder}/ \
            --draw_for_debug=0 --affine_bias=0 --eval_interval_epoch=1 \
            --zero_pad=1 --bos_pad=1 \
            --filename_prefix=processed. &> logging_200_${running_mode}.txt;;
    3)
    echo 'test_given_fullpath'
    if [ -z $2 ] ; then
        echo 'must given epoch idx if testing fiven fullpath'
    else
        export CUDA_VISLBLE_DEVICES=1
        epoch=$2
        python transformer_overall.py --running_mode=test --data_dir=${DATA_DIR} \
        --src_language=${src_language} --tgt_language=${tgt_language} --test_batch_size=32 --beam_width=${beam_size} --alpha=0.6 \
        --model_dir=/space/shr/transformer_wpm/log_dir/${src_language}_${tgt_language}.bsize${BATCH_SIZE}.epoch40.lr_c2warm16000/my-model.epoch${epoch} \
        --filename_prefix=processed.${ENCODER}. --log_disk_dir=${LOG_DISK_DIR}
    fi ;;

    4)
    echo 'load from pytorch model'
    export CUDA_VISIBLE_DEVICES=0
    python transformer_overall.py --running_mode=test --data_dir=${DATA_DIR} --filename_prefix=processed.${ENCODER}. \
        --src_language=${src_language} --tgt_language=${tgt_language} --test_batch_size=2 --beam_width=${beam_size} --alpha=0.6 \
        --load_from_pytorch=1 \
        --model_dir=/home/hzt/shr/transformer_pytorch/temp/run_en_vi/models/ \
        --model_filename=ckpt_from_pytorch.p \
        --log_disk_dir=${LOG_DISK_DIR} --debug=1 &> test_debug.txt;;
esac
