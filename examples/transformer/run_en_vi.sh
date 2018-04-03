export CUDA_VISIBLE_DEVICES=1
MAX_EPOCH=40
DATA_DIR='/home/shr/others_repo/Attention_is_All_You_Need/data/en_vi/data'
# python transformer.py --max_epoch=40 --data_dir=${DATA_DIR} --src_language=en --tgt_language=vi &&

python eval_beam.py --data_dir=${DATA_DIR} --src_language=en --tgt_language=vi --test_batch_size=10 --beam_width=5
