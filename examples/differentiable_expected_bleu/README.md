# Seq2seq Model #

This example builds an attentional seq2seq model for machine translation trained with Differentiable Expected BLEU (DEBLEU) and Teacher Mask. See https://openreview.net/pdf?id=S1x2aiRqFX for the implemented paper.

### Dataset ###

  * iwslt14: The benchmark [IWSLT2014](https://sites.google.com/site/iwsltevaluation2014/home) (de-en) machine translation dataset. 

Download the data with the following cmds:

```
python prepare_data.py --data de-en
```

### Train the model ###

Train the model with the following cmd:

```
python differentiable_expected_bleu.py --config_model config_model --config_data config_iwslt14_de-en --config_train config_train_iwslt14_de-en --pretrain_epochs 8
```

Here:
  * `--config_model` specifies the model config. Note not to include the `.py` suffix.
  * `--config_data` specifies the data config.
  * `--config_train` specifies the training config.
  * `--pretrain_epochs` specifies the number of epochs to pretrain with cross-entropy loss.

[config_model.py](./config_model.py) specifies a single-layer seq2seq model with Luong attention and bi-directional RNN encoder. Hyperparameters taking default values can be omitted from the config file. 

## Results ##

On the IWSLT14 dataset, the model achieves `BLEU=25.35` after annealed all masks, while the cross-entropy trained model achieves `BLEU=24.57`.
