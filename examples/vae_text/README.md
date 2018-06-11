# Variational Autoencoder for Text Generation

This example builds a VAE for text generation, with LSTM as encoder and LSTM or Transformer as decoder.



## Usage

Training can be performed with the following command:

```shell
python vae_train.py --config config_trans --dataset ptb
```

Here:

* `--config` specifies the config file to use

`config_trans.py` specifies the data paths and hyperparameters when transformer is the decoder, and `config_lstm.py` specifies the configuration when LSTM is the decoder.

## Results

|Dataset    |Metrics   | VAE-LSTM |VAE-Transformer |
|---------------|-------------|----------------|------------------------|
|Yahoo | Test PPL<br>Test NLL | 68.31<br>337.36 |61.26<br>328.67|
|PTB | Test PPL<br>Test NLL | 105.27<br>102.06 | 102.46<br>101.46 |

Yahoo dataset is from [(Yang, et. al.) Improved Variational Autoencoders for Text Modeling using Dilated Convolutions](https://arxiv.org/abs/1702.08139).