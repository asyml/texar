import os
import argparse
import codecs
import importlib
import tensorflow as tf
import texar as tx
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

parser = argparse.ArgumentParser(description='prepare data')
parser.add_argument('--dataset', type=str, default='ptb',
                    help='dataset to prepare')
parser.add_argument('--data_path', type=str, default='./',
                    help="Directory containing coco. If not exists, "
                    "the directory will be created, and the data "
                    "will be downloaded.")
parser.add_argument('--config', type=str, default='config_ptb_small',
                    help='The config to use.')
args = parser.parse_args()

config = importlib.import_module(args.config)


def prepare_data(args, config, train_path):
    """Downloads the PTB or Yahoo dataset
    """
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)

    ptb_url = 'https://jxhe.github.io/download/ptb_data.tgz'
    coco_url = 'https://VegB.github.io/downloads/coco_data.tgz'

    data_path = args.data_path

    if not tf.gfile.Exists(train_path):
        url = ptb_url if args.dataset == 'ptb' else coco_url
        tx.data.maybe_download(url, data_path, extract=True)
        os.remove('%s_data.tgz' % args.dataset)


def _get_candidate(file_path):
    with open(file_path, 'r') as fin:
        data = fin.readlines()
        candidate = []
        for line in data:
            candidate.extend(line.strip().split())
    return candidate


def _get_reference(file_path):
    with open(file_path, 'r') as fin:
        data = fin.readlines()
    reference = []
    for line in data:
        reference.extend(line.strip().split())
    reference = [reference]
    return reference


def _get_bleu(reference_path, candidate_path):
    reference = _get_reference(reference_path)
    candidate = _get_candidate(candidate_path)
    bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0),
                          smoothing_function=SmoothingFunction().method4)
    bleu2 = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0),
                          smoothing_function=SmoothingFunction().method4)
    bleu3 = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0),
                          smoothing_function=SmoothingFunction().method4)
    bleu4 = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1),
                          smoothing_function=SmoothingFunction().method4)
    return [bleu1, bleu2, bleu3, bleu4]


def calculate_bleu(config, epoch, inference_list):
    outputs_filename = config.log_dir + 'epoch%d.txt' % epoch
    with codecs.open(outputs_filename, 'w+', 'utf-8') as fout:
        for inf in inference_list:
            fout.write(' '.join(inf) + '\n')
    bleu1, bleu2, bleu3, bleu4 = _get_bleu(
        reference_path=config.train_data_hparams['dataset']['files'],
        candidate_path=outputs_filename)
    buf_train = "epoch %d BLEU1~4 on train dataset:\n%f\n%f\n%f\n%f\n\n" % \
                (epoch, bleu1, bleu2, bleu3, bleu4)
    bleu1, bleu2, bleu3, bleu4 = _get_bleu(
        reference_path=config.test_data_hparams['dataset']['files'],
        candidate_path=outputs_filename)
    buf_test = "epoch %d BLEU1~4 on test dataset:\n%f\n%f\n%f\n%f\n\n" % \
               (epoch, bleu1, bleu2, bleu3, bleu4)

    return buf_train, buf_test


if __name__ == '__main__':
    prepare_data(args, config, config.train_data_hparams['dataset']['files'])
