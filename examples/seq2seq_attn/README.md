# Seq2seq Model #

This example builds an attentional seq2seq model for machine translation.

## Usage ##

### Dataset ###

Two example datasets are provided:

  * toy_copy: A small toy autoencoding dataset from [TF Seq2seq toolkit](https://github.com/google/seq2seq/tree/2500c26add91b079ca00cf1f091db5a99ddab9ae).
  * iwslt14: The benchmark [IWSLT2014](https://sites.google.com/site/iwsltevaluation2014/home) (de-en) machine translation dataset, following [(Ranzato et al., 2015)](https://arxiv.org/pdf/1511.06732.pdf) for data pre-processing.

Download the data with the following commands:

```
python prepare_data.py --data toy_copy
python prepare_data.py --data iwslt14
```

### Train the model ###

Train the model with the following command:

```
python seq2seq_attn.py --config-model config_model --config-data config_toy_copy
```

Here:

  * `--config-model` specifies the model config. Note not to include the `.py` suffix.
  * `--config-data` specifies the data config.

[config_model.py](./config_model.py) specifies a single-layer seq2seq model with Luong attention and bi-directional RNN encoder. Hyperparameters taking default values can be omitted from the config file. 

For demonstration purpose, [config_model_full.py](./config_model_full.py) gives all possible hyperparameters for the model. The two config files will lead to the same model.

## Results ##

On the IWSLT14 dataset, using original target texts as reference (no  `<UNK>`  in the reference), the model achieves `BLEU = 26.44 ± 0.18` .

