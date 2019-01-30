# Transformer for Machine Translation #

This is an implementation of the Transformer model described in [Vaswani, Ashish, et al. "Attention is all you need."](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf).

[Quick Start](https://github.com/asyml/texar/tree/master/examples/transformer#quick-start): Prerequisites & use on machine translation datasets

[Run Your Customized Experiments](https://github.com/asyml/texar/tree/master/examples/transformer#run-your-customized-experiments): Hands-on tutorial of data preparation, configuration, and model training/test

## Quick Start ##

### Prerequisites ###

Run the following cmd to install necessary packages for the example: 
```bash
pip install -r requirements.txt
```

### Datasets ###

Two example datasets are provided:
- IWSLT'15 **EN-VI** for English-Vietnamese translation
- WMT'14 **EN-DE** for English-German translation

Download and pre-process the **IWSLT'15 EN-VI** data with the following cmds: 
```bash
sh scripts/iwslt15_en_vi.sh 
sh preprocess_data.sh spm en vi
```
By default, the downloaded dataset is in `./data/en_vi`. 
As with the [official implementation](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py), `spm` (`sentencepiece`) encoding is used to encode the raw text as data pre-processing. The encoded data is by default in `./temp/run_en_vi_spm`. 

For the **WMT'14 EN-DE** data, download and pre-process with:
```bash
sh scripts/wmt14_en_de.sh
sh preprocess_data.sh bpe en de
```

By default, the downloaded dataset is in `./data/en_de`.
Note that for this dataset, `bpe` encoding (Byte pair encoding) is used instead. The encoded data is by default in `./temp/run_en_de_bpe`. 

### Train and evaluate the model ###

Train the model with the cmd:
```bash
python transformer_main.py --run_mode=train_and_evaluate --config_model=config_model --config_data=config_iwslt15
```
* Specify `--model_dir` to dump model checkpoints, training logs, and tensorboard summaries to a desired directory. By default it is set to `./outputs`. 
* Specifying `--model_dir` will also restore the latest model checkpoint under the directory, if any checkpoint is there.
* Specify `--config_data=config_wmt14` to train on the WMT'14 data.

### Test a trained model ###

To only evaluate a model checkpoint without training, first load the checkpoint and generate samples: 
```bash
python transformer_main.py --run_mode=test --config_data=config_iwslt15 --model_dir=./outputs
```
The latest checkpoint in `./outputs` is used. Generated samples are in the file `./outputs/test.output.hyp`, and reference sentences are in the file `./outputs/test.output.ref` 

Next, decode the samples with respective decoder, and evaluate with `bleu_tool`:
```bash
../../bin/utils/spm_decode --infile ./outputs/test.output.hyp --outfile temp/test.output.spm --model temp/run_en_vi_spm/data/spm-codes.32000.model --input_format=piece 

python bleu_tool.py --reference=data/en_vi/test.vi --translation=temp/test.output.spm
```

For WMT'14, the corresponding cmds are:
```bash
# Loads model and generates samples
python transformer_main.py --run_mode=test --config_data=config_wmt14 --model_dir=./outputs

# BPE decoding
cat outputs/test.output.hyp | sed -E 's/(@@ )|(@@ ?$)//g' > temp/test.output.bpe

# Evaluates BLEU
python bleu_tool.py --reference=data/en_de/test.de --translation=temp/test.output.bpe
```

### Results

* On IWSLT'15, the implementation achieves around `BLEU_cased=28.54` and `BLEU_uncased=29.30` (by [bleu_tool.py](./bleu_tool.py)), which are comparable to the base_single_gpu results by the [official implementation](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py) (`28.12` and `28.97`, respectively, as reported [here](https://github.com/tensorflow/tensor2tensor/pull/611)).

* On WMT'14, the implementation achieves around `BLEU_cased=25.12` (setting: base_single_gpu, batch_size=3072).


### Example training log

```
12:02:02,686:INFO:step:500 loss: 7.3735
12:04:20,035:INFO:step:1000 loss:6.1502
12:06:37,550:INFO:step:1500 loss:5.4877
```
Using an Nvidia GTX 1080Ti, the model usually converges within 5 hours (~15 epochs) on IWSLT'15.

---

## Run Your Customized Experiments

Here is an hands-on tutorial on running Transformer with your own customized dataset.

### 1. Prepare raw data

Create a data directory and put the raw data in the directory. To be compatible with the data preprocessing in the next step, you may follow the convention below:

* The data directory should be named as `data/${src}_${tgt}/`. Take the data downloaded with `scripts/iwslt15_en_vi.sh` for example, the data directory is `data/en_vi`.
* The raw data should have 6 files, which contain source and target sentences of training/dev/test sets, respectively. In the `iwslt15_en_vi` example, `data/en_vi/train.en` contains the source sentences of the training set, where each line is a sentence. Other files are `train.vi`, `dev.en`, `dev.vi`, `test.en`, `test.vi`. 

### 2. Preprocess the data

To obtain the processed dataset, run
```bash
preprocess_data.sh ${encoder} ${src} ${tgt} ${vocab_size} ${max_seq_length}
```
where

* The `encoder` parameter can be `bpe`(byte pairwise encoding), `spm` (sentence piece encoding), or
`raw`(no subword encoding).
* `vocab_size` is optional. The default is 32000. 
  - At this point, this parameter is used only when `encoder` is set to `bpe` or `spm`. For `raw` encoding, you'd have to truncate the vocabulary by yourself.
  - For `spm` encoding, the preprocessing may fail (due to the Python sentencepiece module) if `vocab_size` is too large. So you may want to try smaller `vocab_size` if it happens. 
* `max_seq_length` is optional. The default is 70.

In the `iwslt15_en_vi` example, the cmd is `sh preprocess_data.sh spm en vi`.

By default, the preprocessed data are dumped under `temp/run_${src}_${tgt}_${encoder}`. In the `iwslt15_en_vi` example, the directory is `temp/run_en_vi_spm`.

If you choose to use `raw` encoding method, notice that:

- By default, the word embedding layer is built with the combination of source vocabulary and target vocabulary. For example, if the source vocabulary is of size 3K and the target vocabulary of size 3K and there is no overlap between the two vocabularies, then the final vocabulary used in the model is of size 6K.
- By default, the final output layer of transformer decoder (hidden_state -> logits) shares the parameters with the word embedding layer.

### 3. Specify data and model configuration

Customize the Python configuration files to config the model and data.

Please refer to the example configuration files `config_model.py` for model configuration and `config_iwslt15.py` for data configuration.

### 4. Train the model

Train the model with the following cmd:
```bash
python transformer_main.py --run_mode=train_and_evaluate --config_model=custom_config_model --config_data=custom_config_data
```
where the model and data configuration files are `custom_config_model.py` and `custom_config_data.py`, respectively.

Outputs such as model checkpoints are by default under `outputs/`.

### 5. Test the model

Test with the following cmd:
```bash
python transformer_main.py --run_mode=test --config_data=custom_config_data --model_dir=./outputs
```

Generated samples on the test set are in `outputs/test.output.hyp`, and reference sentences are in `outputs/test.output.ref`. If you've used `bpe` or `spm` encoding in the data preprocessing step, the text in these files are in the respective encoding too. To decode, use the respective cmd:
```bash
# BPE decoding
cat outputs/test.output.hyp | sed -E 's/(@@ )|(@@ ?$)//g' > temp/test.output.hyp.final

# SPM decoding (take `iwslt15_en_vi` for example)
../../bin/utils/spm_decode --infile ./outputs/test.output.hyp --outfile temp/test.output.hyp.final --model temp/run_en_vi_spm/data/spm-codes.32000.model --input_format=piece 
```

Finally, to evaluate the BLEU score against the ground truth on the test set:
```bash
python bleu_tool.py --reference=you_reference_file --translation=temp/test.output.hyp.final
```
E.g., in the `iwslt15_en_vi` example, with `--reference=data/en_vi/test.vi`
