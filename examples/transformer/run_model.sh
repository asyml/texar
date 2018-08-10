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

encoder=spm
DATA_DIR="./temp/run_${src_language}_${tgt_language}_${encoder}/data/"
hparams_set=$1
LOG_DISK_DIR=/space/hzt/shr/transformer_${encoder}/
running_mode=$2
echo "mode:${running_mode}"

if [ ${running_mode}x = "test"x ] ; then
    if [ ${5}x = "max"x ]; then
        model_dir="max"
    else
        model_dir="default"
    fi
fi

case ${hparams_set} in
    1)
    echo 'running the model according to tensor2tensor hparams on small dataset'
    echo "mode ${running_mode} ${src_language}-${tgt_language} "
    logging_filename=logging_${hparams_set}_${src_language}_${tgt_language}_${running_mode}.txt
    echo "logging file:${logging_filename}"
    python transformer_overall.py --mode=${running_mode} --max_train_epoch=70 --max_training_steps=125000 \
        --pre_encoding=${encoder} \
        --data_dir=${DATA_DIR}  --model_dir=${model_dir}\
        --src_language=${src_language} --tgt_language=${tgt_language} \
        --batch_size=2048 --test_batch_size=64 \
        --beam_width=5 --alpha=0.6 \
        --log_disk_dir=/space/hzt/shr/transformer_${encoder}/ \
        --draw_for_debug=0 --affine_bias=0 --eval_interval_epoch=1 \
        --zero_pad=1 --bos_pad=0 --eval_steps=2000 \
        --filename_prefix=processed. &> ${logging_filename};;

    2)
        echo 'running the model with bigger batch_size and training steps'
        echo 'only support en and de language for now'
        src_language=en
        tgt_language=de
        logging_filename=logging_${hparams_set}_${running_mode}.txt
        echo "mode ${running_mode} ${src_language}-${tgt_language}"
        python transformer_overall.py --mode=${running_mode} --max_train_epoch=70\
            --pre_encoding=${encoder} --data_dir=${DATA_DIR} \
            --src_language=${src_language} --tgt_language=${tgt_language} \
            --batch_size=3072 --test_batch_size=32 --max_training_steps=500000\
            --beam_width=5 --alpha=0.6 \
            --log_disk_dir=/space/shr/transformer_${encoder}/ \
            --draw_for_debug=0 --affine_bias=0 --eval_interval_epoch=1 \
            --zero_pad=1 --bos_pad=1 \
            --filename_prefix=processed. &> ${logging_filename};;
    4)
    echo 'load from pytorch model, this feature is not mature.'
    src_language=en
    tgt_language=vi
    encoder=bpe
    DATA_DIR="./temp/run_${src_language}_${tgt_language}_${encoder}/data/"
    python transformer_overall.py --running_mode=test --data_dir=${DATA_DIR} --filename_prefix=processed. \
        --pre_encoding=${encoder} \
        --src_language=${src_language} --tgt_language=${tgt_language} --test_batch_size=2 --beam_width=5 --alpha=0.6 \
        --load_from_pytorch=1 \
        --model_dir=/home/hzt/shr/transformer_pytorch/temp/run_en_vi/models/ \
        --model_filename=ckpt_from_pytorch.p \
        --log_disk_dir=${LOG_DISK_DIR} --debug=1 &> test_debug.txt;;
esac
