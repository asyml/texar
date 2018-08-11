# SeqGAN for Text Generation

This example is an implementation of [(Yu et al.) SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/pdf/1609.05473.pdf), with a language model as the generator and an RNN-based classifier as the discriminator.

Model structure and parameter settings are in line with the [official implementation](https://github.com/geek-ai/Texygen) of SeqGAN, except that we replace the MC-Tree rollout strategy with token-level reward by the RNN discriminator, which is simpler and provides competitive performance.

Experiments are performed on two datasets:
* The standard [PTB dataset](https://corochann.com/penn-tree-bank-ptb-dataset-introduction-1456.html)
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
* `--data_path` is the directory to store the downloaded dataset. Default is './'.
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

|    |Texar - SeqGAN   | Official - SeqGAN | Texar - MLE | Official - MLE |
|---------------|-------------|----------------|-------------|----------------|
|BLEU-1 | 0.5663 (0.7446) | 0.5709 (0.7192) | 0.6066 (0.8274) | 0.5730 (0.7450) |
|BLEU-2 | 0.2887 (0.5322) | 0.2657 (0.4465) | 0.2941 (0.5791) | 0.2856 (0.5242) |
|BLEU-3 | 0.1209 (0.2979) | 0.0981 (0.2202) | 0.1194 (0.3099) | 0.1190 (0.2810) |
|BLEU-4 | 0.0424 (0.1324) | 0.0287 (0.0828) | 0.0414 (0.1330) | 0.0417 (0.1212)|

### PTB

|config|train|valid|test| Official - test |
|---|---|---|---|---|
|small|26.8470|55.6829|53.3579||
|medium|8.4457|15.7546|15.4920||
|large|6.633471|16.559490|16.190385||

## Training Log

During training, loss and BLEU score are recorded in the log directory. Here, we provide sample log output when training on the  `coco` dataset.

### Training loss
Training loss will be recorded in coco_log/log.txt.
```text
G pretrain epoch   0, step 1: train_ppl: 81.639235
G pretrain epoch   1, step 1: train_ppl: 9.845531
G pretrain epoch   2, step 1: train_ppl: 7.581516
...
G pretrain epoch  78, step 1: train_ppl: 3.753437
G pretrain epoch  79, step 1: train_ppl: 3.711618
D pretrain epoch   0, step 0: dis_total_loss: 16.657263, r_loss: 8.789272, f_loss: 7.867990
D pretrain epoch   1, step 0: dis_total_loss: 3.317280, r_loss: 1.379951, f_loss: 1.937329
D pretrain epoch   2, step 0: dis_total_loss: 1.798969, r_loss: 0.681685, f_loss: 1.117284
...
D pretrain epoch  78, step 0: dis_total_loss: 0.000319, r_loss: 0.000009, f_loss: 0.000310
D pretrain epoch  79, step 0: dis_total_loss: 0.000097, r_loss: 0.000009, f_loss: 0.000088
G update   epoch  80, step 1: mean_reward: -56.315876, expect_reward_loss:-56.315876, update_loss: 9194.217773
D update   epoch  80, step 0: dis_total_loss: 0.000091, r_loss: 0.000008, f_loss: 0.000083
G update   epoch  81, step 1: mean_reward: -56.507019, expect_reward_loss:-56.507019, update_loss: 10523.346680
D update   epoch  81, step 0: dis_total_loss: 0.000230, r_loss: 0.000008, f_loss: 0.000222
...
G update   epoch 178, step 1: mean_reward: -58.171032, expect_reward_loss:-58.171032, update_loss: 15077.129883
D update   epoch 178, step 0: dis_total_loss: 0.000073, r_loss: 0.000003, f_loss: 0.000070
G update   epoch 179, step 1: mean_reward: -58.190083, expect_reward_loss:-58.190083, update_loss: 14430.581055
D update   epoch 179, step 0: dis_total_loss: 0.000019, r_loss: 0.000003, f_loss: 0.000016
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
