export CUDA_VISIBLE_DEVICES=''

MAX_EPOCH=40
DATA_DIR='/home/hzt/shr/t2t_data/wmt_ende'
BATCH_SIZE=3072
MAX_TRAIN_STEP=500000
LOG_DISK_DIR='/space/shr/transformer_log/'
MODEL_DIR='/space/shr/transformer_log/log_dir/bsize3072.step500000.lr_c2warm16000'

#train while eval
#python transformer.py --max_epoch=40 --data_dir=${DATA_DIR} --src_language=en --tgt_language=de --batch_size=3072 --max_training_steps=500000
#

#train while eval
#python transformer_overall.py --running_mode=test --data_dir=${DATA_DIR} --src_language=en --tgt_language=de --batch_size=3072 --max_training_steps=500000 --test_batch_size=32 --beam_width=5 --alpha=0.6 --log_disk_dir=${LOG_DISK_DIR}

#only test
python transformer_overall.py --running_mode=test \
    --data_dir=${DATA_DIR} --src_language=en --tgt_langauge=de \
    --test_batch_size=32 --beam_width=5 --alpha=0.6 \
    --model_dir=${MODEL_DIR} --verbose=1 \
    --decode_from_file='/tmp/t2t_datagen/newstest2014.tok.bpe.32000.en'
