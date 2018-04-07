export CUDA_VISIBLE_DEVICES=2
MAX_EPOCH=40
DATA_DIR='/home/shr/others_repo/Attention_is_All_You_Need/data/en_vi/data'

# train+ source equals about 3000, so I guess src, tgt seperately be ~1500, so batch_size should be 2048
# watch whether it can converge

#python transformer.py --max_epoch=40 --data_dir=${DATA_DIR} --src_language=en --tgt_language=vi \
#    --batch_size=2048 &&
python eval_beam.py --data_dir=${DATA_DIR} --src_language=en --tgt_language=vi --test_batch_size=10 --beam_width=5
