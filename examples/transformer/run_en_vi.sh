export CUDA_VISIBLE_DEVICES=0

MAX_EPOCH=100
MAX_TRAINING_STEPS=125000
BATCH_SIZE=2048
ENCODER=wpm
# generally it will be about 70 epoches in en-vi dataset
#DATA_DIR='/home/shr/others_repo/Attention_is_All_You_Need/data/en_vi/data'
DATA_DIR=/home/hzt/shr/others_repo/Attention_is_All_You_Need/data/en_vi/data
#LOG_DISK_DIR='/home2/shr/transformer/'
LOG_DISK_DIR=/space/shr/transformer_${ENCODER}/

#python transformer.py --max_train_epoch=40 --data_dir=${DATA_DIR} --src_language=en --tgt_language=de --batch_size=2048 --save_checkpoint_interval=2000
#python eval_beam.py --data_dir=${DATA_DIR} --src_language=en --tgt_language=de --beam_width=5 --alpha-0.6

#for bpe
#python transformer_overall.py --running_mode=train --max_train_epoch=${MAX_EPOCH} --max_training_steps=${MAX_TRAINING_STEPS} --data_dir=${DATA_DIR} --src_language=en --tgt_language=vi --batch_size=${BATCH_SIZE} --test_batch_size=32 --beam_width=4 --alpha=0.6 --save_checkpoint_interval=2000 --log_disk_dir=${LOG_DISK_DIR} --filename_prefix=processed.bpe.

#for wpm
#export CUDA_VISIBLE_DEVICES=0
#python transformer_overall.py --running_mode=train --max_train_epoch=${MAX_EPOCH} --max_training_steps=${MAX_TRAINING_STEPS} --data_dir=${DATA_DIR} --src_language=en --tgt_language=vi --batch_size=${BATCH_SIZE} --test_batch_size=32 --beam_width=4 --alpha=0.6 --save_checkpoint_interval=2000 --log_disk_dir=${LOG_DISK_DIR} --filename_prefix=processed.${ENCODER}.

export CUDA_VISIBLE_DEVICES=1

python transformer_overall.py --running_mode=test --data_dir=${DATA_DIR} --src_language=en --tgt_language=vi --test_batch_size=32 --beam_width=1 --alpha=0.6 --model_dir=/space/shr/transformer_wpm/log_dir/en_vi.bsize2048.epoch100.lr_c2warm16000/ --filename_prefix=processed.${ENCODER}. --log_disk_dir=${LOG_DISK_DIR}
