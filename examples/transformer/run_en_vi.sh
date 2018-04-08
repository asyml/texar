export CUDA_VISIBLE_DEVICES=2
MAX_EPOCH=40
DATA_DIR='/home/shr/others_repo/Attention_is_All_You_Need/data/en_vi/data'
BATCH_SIZE=1536

# train+ source equals about 3000, so I guess src, tgt seperately be ~1500, so batch_size should be 1536, I have also tried 2048
# to check the performance

#python transformer.py --max_train_epoch=40 --data_dir=${DATA_DIR} --src_language=en --tgt_language=vi --batch_size=1536

#python eval_beam.py --data_dir=${DATA_DIR} --src_language=en --tgt_language=vi --test_batch_size=10 --beam_width=5 --alpha=0.6 --max_train_epoch=40 --batch_size=${BATCH_SIZE}
