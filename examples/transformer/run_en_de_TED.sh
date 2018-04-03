export CUDA_VISIBLE_DEVICES=2
MAX_EPOCH=40
DATA_DIR='/home/shr/others_repo/Attention_is_All_You_Need/data/en_de_TED/data'


#python transformer.py --max_epoch=40 --data_dir=${DATA_DIR} --src_language=en --tgt_language=de
python eval_greedy_naive.py --data_dir=${DATA_DIR} --src_language=en --tgt_language=de
