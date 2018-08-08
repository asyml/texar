# Variational Autoencoder for Text Generation

This example builds a VAE for text generation, with LSTM as encoder and LSTM or Transformer as decoder. Training is performed on official PTB data and Yahoo data, respectively. Yahoo dataset is from [(Yang, et. al.) Improved Variational Autoencoders for Text Modeling using Dilated Convolutions](https://arxiv.org/abs/1702.08139), which is created by sampling 100k documents from the original Yahoo Answer data. The average document length is 78 and the vocab size is 200k. 

## Data
The PTB and Yahoo datasets can be downloaded [here]().



## Usage
Training can be performed with the following command:

```shell
python vae_train.py --config config_trans_ptb --dataset ptb
```

Here:

* `--config` specifies the config file to use. We provide the config files used to reproduce the results, where config_lstm_xxx.py includes the configuration for LSTM as the decoder, config_trans_xxx.py includes the configuration for transformer as the decoder. 
* `--dataset` specifies the dataset to use, currently `ptb` and `yahoo` are supported

## Results

|Dataset    |Metrics   | VAE-LSTM |VAE-Transformer |
|---------------|-------------|----------------|------------------------|
|Yahoo | Test PPL<br>Test NLL | 68.31<br>337.36 |59.56<br>326.41|
|PTB | Test PPL<br>Test NLL | 105.48<br>102.10 | 102.53<br>101.48 |

## Reference
* [Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349)

