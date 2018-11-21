# Seq2seq Model #

This example builds an attentional seq2seq model for machine translation trained with Differentiable Expected BLEU (DEBLEU) and Teacher Mask. See https://openreview.net/pdf?id=S1x2aiRqFX for the implemented paper.

### Dataset ###

  * iwslt14: The benchmark [IWSLT2014](https://sites.google.com/site/iwsltevaluation2014/home) (de-en) machine translation dataset. 

Download the data with the following cmds:

```bash
python prepare_data.py --data de-en
```

### Train the model ###

Train the model with the following cmd:

```bash
python differentiable_expected_bleu.py --config_model config_model_medium --config_data config_data_iwslt14_de-en --config_train config_train --expr_name iwslt14_de-en --restore_from "" --reinitialize
```

Here:
  * `--config_model` specifies the model config. Note not to include the `.py` suffix.
  * `--config_data` specifies the data config.
  * `--config_train` specifies the training config.
  * `--expr_name` specifies the experiment name. Used as the directory name to save and restore all information.
  * `--restore_from` specifies the checkpoint path to restore from. If not specified (or an empty string is specified), the latest checkpoint in `expr_name` is restored.
  * `--reinitialize` is a flag indicates whether to reinitialize the state of the optimizers before training and after annealing. Default is enabled.

[config_model_medium.py](./config_model_medium.py) specifies a single-layer seq2seq model with Luong attention and bi-directional RNN encoder.

[config_model_large.py](./config_model_large.py) specifies a seq2seq model with Luong attention, 2-layer bi-directional RNN encoder, single-layer RNN decoder, and a connector between the final state of the encoder and the initial state of the decoder. The size of this model is quite large.

[config_data_iwslt14_de-en.py](./config_data_iwslt14_de-en.py) specifies the IWSLT'14 German-English dataset.

[config_train.py](./config_train.py) specifies the training (including annealing) configs.

## Results ##

On the IWSLT'14 German-English dataset, we ran both configs for 4~5 times. Here are the average BLEU scores attained:

|                       config                       | inference beam size | Cross-Entropy baseline | DEBLEU | improvement |
| :------------------------------------------------: | :-----------------: | :--------------------: | :----: | :---------: |
| [config_model_medium.py](./config_model_medium.py) |          1          |         26.12          | 27.40  |    1.28     |
| [config_model_medium.py](./config_model_medium.py) |          5          |         27.03          | 27.72  |    0.70     |
|  [config_model_large.py](./config_model_large.py)  |          1          |         25.24          | 26.47  |    1.23     |
|  [config_model_large.py](./config_model_large.py)  |          5          |         26.33          | 26.87  |    0.54     |
