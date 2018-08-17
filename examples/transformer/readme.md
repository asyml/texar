This example gives an implementation of [Vaswani, Ashish, et al. "Attention is all you need."](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf), based on self-attention mechanism for language understanding.

Here we give the experimental results on two dataset, while it's safe for you to try on your datasets.

- EN-VI: IWSLT'15 English-Vietnamese dataset.
- EN-DE: WMT14 English-German dataset.

# Prerequisites

In addition to installing the dependency for texar library, you need to
run `pip install -r requirements` to install the dependencies for transformer translation model.

# For EN-VI dataset.

The task is IWSLT'15 English-Vietnamese dataset. For more information, please refer to https://nlp.stanford.edu/projects/nmt/

## Obtain the dataset
```
mkdir data/en_vi
cp your train.en train.vi dev.en dev.vi test.en test.vi into this directory.
```
Feel free to try on different datasets as long as they are parallel text corpora and the file paths are set correctly.

## Preprocessing the dataset and generate encoded vocabulary
```
bash preprocess_data.sh en vi
```
By default, we use SentencePiece encoder to keep consistent with tensor2tensor, you could also change the `encoder` variable in `preprocess_data.sh` to `bpe` to use BytePairwise Encoding.

## Training and evaluating the model

```
#train
bash run_model.sh 1 train_and_evaluate en vi

#test
bash run_model 1 test en vi

#evaluate with BLEU score
bash test_output.sh en vi
```

The `1` indicates one hparams set for en-vi task: `max_train_epoch=70 max_training_steps=125000 batch_size=2048 test_batch_size=64 beam_width=5 alpha=0.6 ...`. You need to manually set the `log_disk_dir` parameter to control the output path of the tensorflow logging file. Read the `run_model.sh` for more details.

## Result

You could get ~28.4 BLEU_cased and ~28.97 BLEU_uncased with our implementation. With tensor2tensor, the results are 28.12 and 28.97 claimed in https://github.com/tensorflow/tensor2tensor/pull/611.

## Training log sample:

```
11:59:46,112:INFO:step:0 source:(44, 63) targets:(44, 67) loss:10.917707
12:02:02,686:INFO:step:500 source:(47, 62) targets:(47, 62) loss:7.2396936
12:04:20,035:INFO:step:1000 source:(48, 61) targets:(48, 57) loss:6.0277987
12:06:37,550:INFO:step:1500 source:(43, 52) targets:(43, 69) loss:5.589434
```

The model can converge within 5 hours (~15 epochs).

# We also give a sample script for wmt14 en-de task with Byte Pairwise Encoding here.

## Obtain the dataset
```
#change the DOWNLOADED_DATA_DIR in the wmt14_en_de.sh to your own path.
bash wmt14_en_de.sh
```
You will obtain the dataset in the `./data/en_de/` directory

## Preprocessing the dataset and generate encoded vocabulary
```
#Modify the `encoder` in `preprocess_data.sh` to `bpe` to use Byte Pairwise Encoding.
bash preprocess_data.sh en de
```
You will obtain the processed dataset in `./temp/data/run_en_de_bpe/data/` directory

## Training the model

```
#Modify the `encoder` in `run_model.sh` to `bpe`

bash run_model 2 train_and_evaluate en de
```
Here `2` denotes one hparams set for wmt14 en-de task (model with more
parameters compared to `1` hparams set).

## Test and evaluation
```
bash run_model.sh 2 test en de

#Modify the `encoder` in `test_output.sh` to `bpe`
bash test_output.sh en de
```

## Result

You could get ~25.12 BLEU_cased with our implementation. With tensor2tensor, the running result is 25.35 with the setting of base_single_gpu and 3072 batch size.


