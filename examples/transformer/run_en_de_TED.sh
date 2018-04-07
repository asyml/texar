export CUDA_VISIBLE_DEVICES=3
MAX_EPOCH=40
DATA_DIR='/home/shr/others_repo/Attention_is_All_You_Need/data/en_de_TED/data'

python transformer.py --max_train_epoch=40 --data_dir=${DATA_DIR} --src_language=en --tgt_language=de --batch_size=2048 --save_checkpoint_interval=2000
#python eval_beam.py --data_dir=${DATA_DIR} --src_language=en --tgt_language=de --beam_width=5 --alpha-0.6

#python transformer_overall.py --max_train_epoch=40 --data_dir=${DATA_DIR} --src_language=en --tgt_language=de --batch_size=2048 --test_batch_size=10 --beam_width=5 --save_checkpoint_interval=2000
