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
bash scripts/iwslt15_en_vi.sh
```
Feel free to try on different datasets as long as they are parallel text corpora and the file paths are set correctly.

## Preprocessing the dataset and generate encoded vocabulary

```
bash preprocess_data.sh spm en vi
```

By default, we use SentencePiece encoder to keep consistent with tensor2tensor.

## Training and evaluating the model

```
#train
#LOG_DISK_DIR=YOUR_CUSTOM_DIR/en_vi/
#change the LOG_DISK_DIR to your own path to save tensorboard logging information and trained model.
python transformer_overall.py --run_mode=train_and_evaluate --config_data=config_iwslt14 --log_dir=${LOG_DISK_DIR}

#test
python transformer_overall.py --run_mode=test --config_data=config_iwslt14 --log_dir=${LOG_DISK_DIR}

# The decoded file path will be in $LOG_DISK_DIR/test.output
#evaluate with BLEU score
export PATH=$PATH:../../bin/utils/
test.outputs=${LOG_DISK_DIR}/test.output
../../bin/utils/spm_decode --model temp/run_en_vi_spm/data/spm-codes.32000.model --input_format=piece --infile ${test.outputs} --outfile test.out

python bleu_tool.py --reference=data/en_vi/test.vi --translation=test.out
```

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
bash scripts/wmt14_en_de.sh
```

You will obtain the dataset in the `./data/en_de/` directory

## Preprocessing the dataset and generate encoded vocabulary
```
bash preprocess_data.sh bpe en de
```

You will obtain the BPE-encoded dataset in `./temp/data/run_en_de_bpe/data/` directory

## Training the model

```
#LOG_DISK_DIR=YOUR_CUSTOM_DIR/en_de/
python transformer_overall.py --mode=train_and_evaluate --config_data=config_wmt14 --log_dir=$LOG_DISK_DIR --wbatchsize=3072
```

## Test and evaluation
```
python transformer_overall.py --run_mode=test --config_data=config_wmt14 --log_dir=$LOG_DISK_DIR

TEST_OUTPUT=${LOG_DISK_DIR}/en_de/test.output
cat ${TEST_OUTPUT} | sed -E 's/(@@ )|(@@ ?$)//g' > test.out

python bleu_tool.py --reference=data/en_de/test.de --translation=test.out
```

## Result

You could get ~25.12 BLEU_cased with our implementation. With tensor2tensor, the running result is 25.35 with the setting of base_single_gpu and 3072 batch size.


