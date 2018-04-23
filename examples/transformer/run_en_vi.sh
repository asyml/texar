mode=$1


MAX_TRAINING_STEPS=125000
MAX_EPOCH=40
BATCH_SIZE=3072
ENCODER=bpe
beam_size=5

#MAX_EPOCH=70
#BATCH_SIZE=2048
#ENCODER=wpm
#beam_size=4
# generally it will be about 70 epoches in en-vi dataset
#DATA_DIR='/home/shr/others_repo/Attention_is_All_You_Need/data/en_vi/data'
DATA_DIR=/home/hzt/shr/others_repo/Attention_is_All_You_Need/data/en_vi/data
#LOG_DISK_DIR='/home2/shr/transformer/'
LOG_DISK_DIR=/space/shr/transformer_${ENCODER}/
#beam_size=5
mode=$1
case $mode in
    1)
    echo 'training the model...'
    #export CUDA_VISIBLE_DEVICES=0
    python transformer_overall.py --running_mode=train_and_evaluate --max_train_epoch=${MAX_EPOCH} --max_training_steps=${MAX_TRAINING_STEPS}\
        --data_dir=${DATA_DIR} --src_language=en --tgt_language=vi --batch_size=${BATCH_SIZE} --test_batch_size=32 \
        --beam_width=${beam_size} --alpha=0.6 --save_checkpoint_interval=2000 --log_disk_dir=${LOG_DISK_DIR} \
        --draw_for_debug=1 \
        --filename_prefix=processed.${ENCODER}. ;;
    2)
    echo 'test_given_path'
    export CUDA_VISIBLE_DEVICES=1
        python transformer_overall.py --running_mode=test --data_dir=${DATA_DIR} --batch_size=${BATCH_SIZE} \
        --src_language=en --tgt_language=vi --test_batch_size=32 --beam_width=${beam_size} --alpha=0.6 \
        --model_dir=/space/shr/transformer_${ENCODER}/log_dir/en_vi.bsize${BATCH_SIZE}.epoch${MAX_EPOCH}.lr_c2warm16000/ \
        --filename_prefix=processed.${ENCODER}. --log_disk_dir=${LOG_DISK_DIR} ;;
    3)
    echo 'test_given_fullpath'
    if [ -z $2 ] ; then
        echo 'must given epoch idx if testing fiven fullpath'
    else
        export CUDA_VISLBLE_DEVICES=1
        python transformer_overall.py --running_mode=test --data_dir=${DATA_DIR} \
        --src_language=en --tgt_language=vi --test_batch_size=32 --beam_width=${beam_size} --alpha=0.6 \
        --model_fullpath=/space/shr/transformer_wpm/log_dir/en_vi.bsize${BATCH_SIZE}.epoch40.lr_c2warm16000/my-model.epoch${epoch} \
        --filename_prefix=processed.${ENCODER}. --log_disk_dir=${LOG_DISK_DIR}
    fi
esac
