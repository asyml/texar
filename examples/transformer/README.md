# Transformer for Machine Translation #

This is an implementation of the Transformer model described in [Vaswani, Ashish, et al. "Attention is all you need."](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf).

## Usage ##

### Prerequisites ###

Run the following cmd to install necessary packages for the example: 
```
pip install -r requirements.txt
```

### Datasets ###

Two example datasets are provided:
- IWSLT'15 **EN-VI** for English-Vietnamese translation
- WMT'14 **EN-DE** for English-German translation

Download and pre-process the **IWSLT'15 EN-VI** data with the following cmds: 
```
sh scripts/iwslt15_en_vi.sh 
sh preprocess_data.sh spm en vi
```
By default, the downloaded dataset is in `./data/en_vi`. 
As with the [official implementation](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py), `spm` (`sentencepiece`) encoding is used to encode the raw text as data pre-processing. The encoded data is by default in `./temp/run_en_vi_spm`. 

For the **WMT'14 EN-DE** data, download and pre-process with:
```
sh scripts/wmt14_en_de.sh
sh preprocess_data.sh bpe en de
```

By default, the downloaded dataset is in `./data/en_de`.
Note that for this dataset, `bpe` encoding (Byte pair encoding) is used instead. The encoded data is by default in `./temp/run_en_de_bpe`. 

### Train and evaluate the model ###

Train the model with the cmd:
```
python transformer_main.py --run_mode=train_and_evaluate --config_model=config_model --config_data=config_iwslt15
```
* You can also specify `--model_dir` to dump model checkpoints, training logs, and tensorboard summaries to a desired directory. By default it is set to `./outputs`. 
* Specify `--config_data=config_wmt14` to train on the WMT'14 data.

### Test a trained model ###

To only evaluate a model checkpoint without training, first load the checkpoint and generate samples: 
```
python transformer_main.py --run_mode=test --config_data=config_iwslt15 --model_dir=./outputs
```
The latest checkpoint in `./outputs` is used. Generated samples are in the file `./outputs/test.output`. 

Next, decode the samples with respective decoder, and evaluate with `bleu_tool`:
```
../../bin/utils/spm_decode --infile ./outputs/test.output.src --outfile temp/test.output.spm --model temp/run_en_vi_spm/data/spm-codes.32000.model --input_format=piece 

python bleu_tool.py --reference=data/en_vi/test.vi --translation=temp/test.output.spm
```

For WMT'14, the corresponding cmds are:
```
# Loads model and generates samples
python transformer_main.py --run_mode=test --config_data=config_wmt14 --log_dir=./outputs

# BPE decoding
cat outputs/test.output.src | sed -E 's/(@@ )|(@@ ?$)//g' > temp/test.output.bpe

# Evaluates BLEU
python bleu_tool.py --reference=data/en_de/test.de --translation=temp/test.output.bpe
```

## Results

* On IWSLT'15, the implementation achieves around `BLEU_cased=28.54` and `BLEU_uncased=29.30` (by [bleu_tool.py](./bleu_tool.py)), which are comparable to the base_single_gpu results by the [official implementation](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py) (`28.12` and `28.97`, respectively, as reported [here](https://github.com/tensorflow/tensor2tensor/pull/611)).

* On WMT'14, the implementation achieves around `BLEU_cased=25.12` (setting: base_single_gpu, batch_size=3072).


## Example training log

```
12:02:02,686:INFO:step:500 loss: 7.3735
12:04:20,035:INFO:step:1000 loss:6.1502
12:06:37,550:INFO:step:1500 loss:5.4877
```
Using an Nvidia GTX 1080Ti, the model usually converges within 5 hours (~15 epochs) on IWSLT'15.

