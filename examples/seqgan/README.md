# SeqGAN for Text Generation

This example is an implementation of [(Yu et al.) SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/pdf/1609.05473.pdf), with a language model as the generator and an RNN-based classifier as the discriminator.

Model architecture and parameter settings are in line with the [official implementation](https://github.com/geek-ai/Texygen) of SeqGAN, except that we replace the MC-Tree rollout strategy with token-level reward by the RNN discriminator, which is simpler and provides competitive performance.

Experiments are performed on two datasets:
* The [PTB dataset](https://corochann.com/penn-tree-bank-ptb-dataset-introduction-1456.html) standard for language modeling
* The [COCO Captions dataset](http://cocodataset.org/#download): with 2K vocabularies and an average sentence length of 25. We use the [data](https://github.com/geek-ai/Texygen/tree/master/data) provided in the official implementation, where train/test datasets contain 10K sentences, respectively.

## Usage

### Dataset
Download datasets with the following cmds respectively:
```shell
python data_utils.py --config config_ptb_small --data_path ./ --dataset ptb
python data_utils.py --config config_coco --data_path ./ --dataset coco
```

Here:
* `--config` specifies config parameters to use. Default is `config_ptb_small`.
* `--data_path` is the directory to store the downloaded dataset. Default is `./`.
* `--dataset` indicates the training dataset. Currently `ptb`(default) and `coco` are supported.

### Train the model

Training on `coco` dataset can be performed with the following command:

```shell
python seqgan_train.py --config config_coco --data_path ./ --dataset coco
```

Here:

`--config`, `--data_path` and `--dataset` should be the same with the flags settings used to download the dataset.

The model will start training and will evaluate perplexity and BLEU score every 10 epochs.

## Results

### COCO Caption

We compare the results of SeqGAN and MLE (maximum likelihood training) provided by our and official implemantations, using the default official parameter settings. Each cell below presents the BLEU scores on both the test set and the training set (in the parentheses). 

We use the standard BLEU function [`texar.tf.evals.sentence_bleu_moses`](https://texar.readthedocs.io/en/latest/code/evals.html#sentence-bleu-moses) to evaluate BLEU scores for both the official and our implementations.

|    |Texar - SeqGAN   | Official - SeqGAN | Texar - MLE | Official - MLE |
|---------------|-------------|----------------|-------------|----------------|
|BLEU-1 | 0.5670 (0.6850) | 0.6260 (0.7900) | 0.7130 (0.9360) | 0.6620 (0.8770) |
|BLEU-2 | 0.3490 (0.5330) | 0.3570 (0.5880) | 0.4510 (0.7590) | 0.3780 (0.6910) |
|BLEU-3 | 0.1940 (0.3480) | 0.1660 (0.3590) | 0.2490 (0.4990) | 0.1790 (0.4470) |
|BLEU-4 | 0.0940 (0.1890) | 0.0710 (0.1800) | 0.1170 (0.2680) | 0.0790 (0.2390)|

### PTB

On PTB data, we use three different hyperparameter configurations which result in models of different sizes.
The perplexity on both the test set and the training set are listed in the following table.

|config|train   |Official - train |test    |  Official - test |
|---   |---     |---              |---     |---               |
|small |28.4790 |53.2289          |58.9798 | 55.7736          |
|medium|16.3243 |9.8919           |37.6558 | 20.8537          |
|large |14.5739 |4.7015           |52.0850 | 39.7949          |

## Training Log

During training, loss and BLEU score are recorded in the log directory. Here, we provide sample log output when training on the  `coco` dataset.

### Training loss
Training loss will be recorded in coco_log/log.txt.
```text
G pretrain epoch   0, step   1: train_ppl: 1781.854030
G pretrain epoch   1, step 201: train_ppl: 10.483647
G pretrain epoch   2, step 401: train_ppl: 7.335757
...
G pretrain epoch  77, step 12201: train_ppl: 3.372638
G pretrain epoch  78, step 12401: train_ppl: 3.534658
D pretrain epoch   0, step   0: dis_total_loss: 27.025223, r_loss: 13.822192, f_loss: 13.203032
D pretrain epoch   1, step   0: dis_total_loss: 26.331108, r_loss: 13.592842, f_loss: 12.738266
D pretrain epoch   2, step   0: dis_total_loss: 27.042515, r_loss: 13.592712, f_loss: 13.449802
...
D pretrain epoch  77, step   0: dis_total_loss: 25.134272, r_loss: 12.660420, f_loss: 12.473851
D pretrain epoch  78, step   0: dis_total_loss: 23.727032, r_loss: 12.822734, f_loss: 10.904298
D pretrain epoch  79, step   0: dis_total_loss: 24.769077, r_loss: 12.733292, f_loss: 12.035786
G train  epoch  80, step 12601: mean_reward: 0.027631, expect_reward_loss:-0.256241, update_loss: -20.670971
D train  epoch  80, step   0: dis_total_loss: 25.222481, r_loss: 12.671371, f_loss: 12.551109
D train  epoch  81, step   0: dis_total_loss: 25.695383, r_loss: 13.037079, f_loss: 12.658304
...
G train  epoch 178, step 22401: mean_reward: 3.409714, expect_reward_loss:-3.474687, update_loss: 733.247009
D train  epoch 178, step   0: dis_total_loss: 24.715553, r_loss: 13.181369, f_loss: 11.534184
D train  epoch 179, step   0: dis_total_loss: 24.572170, r_loss: 13.176209, f_loss: 11.395961
```

### BLEU
BLEU1~BLEU4 scores will be calculated every 10 epochs, the results are written to log_dir/bleu.txt.
```text
...
epoch 170 BLEU1~4 on train dataset:
0.726647
0.530675
0.299362
0.133602

 epoch 170 BLEU1~4 on test dataset:
0.548151
0.283765
0.118528
0.042177
...
```

