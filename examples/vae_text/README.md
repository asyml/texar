# Variational Autoencoder (VAE) for Text Generation

This example builds a VAE for text generation, with an LSTM as encoder and an LSTM or [Transformer](https://arxiv.org/pdf/1706.03762.pdf) as decoder. Training is performed on the official PTB data and Yahoo data, respectively. 

The VAE with LSTM decoder is first decribed in [(Bowman et al., 2015) Generating Sentences from a Continuous Space](https://arxiv.org/pdf/1511.06349.pdf)

The Yahoo dataset is from [(Yang et al., 2017) Improved Variational Autoencoders for Text Modeling using Dilated Convolutions](https://arxiv.org/pdf/1702.08139.pdf), which is created by sampling 100k documents from the original Yahoo Answer data. The average document length is 78 and the vocab size is 200k. 

## Data
The datasets can be downloaded by running:
```shell
python prepare_data.py --data ptb
python prepare_data.py --data yahoo
```

## Usage
Train with the following command:

```shell
python vae_train.py --config config_trans_ptb
```

Here:

* `--config` specifies the config file to use, including model hyperparameters and data paths. We provide 4 config files:
  - [config_lstm_ptb.py](./config_lstm_ptb.py): LSTM decoder, on the PTB data
  - [config_lstm_yahoo.py](./config_lstm_yahoo.py): LSTM decoder, on the Yahoo data
  - [config_trans_ptb.py](./config_trans_ptb.py): Transformer decoder, on the PTB data
  - [config_trans_yahoo.py](./config_trans_yahoo.py): Transformer decoder, on the Yahoo data
  
## Results

|Dataset    |Metrics   | VAE-LSTM |VAE-Transformer |
|---------------|-------------|----------------|------------------------|
|Yahoo | Test PPL<br>Test NLL | 68.11<br>337.13 |59.95<br>326.93|
|PTB | Test PPL<br>Test NLL | 104.61<br>101.92 | 103.68<br>101.72 |

