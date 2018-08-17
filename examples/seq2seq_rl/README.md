# Seq2seq Model with Policy Gradient Training #

This example builds an attentional seq2seq model that is trained with policy gradient and BLEU reward. The example is mainly for demonstration of the Texar sequence Reinforcement Learning APIs. No MLE pre-training is included so the model collapsed very quickly. In practice one would usually pretrain the model with teacher-forcing MLE (e.g., see the example [seq2seq_attn](../seq2seq_attn)) and continue to fine-tune with policy gradient. 

The data and model configs are exact the same as the [MLE seq2seq example](../seq2seq_attn). The only difference is that MLE cross-entropy minimization is replaced with policy gradient training.

The example shows:
  * Use of `texar.agents.SeqPGAgent` for policy gradient sequence generation.
  * Use of the Python-based `texar.evals.sentence/corpus_bleu` for efficient reward computing, and the Moses `texar.evals.sentence/corpus_bleu_moses`
    for standard test set evaluation.
  * Use of `texar.data.FeedableDataIterator` for data feeding and resuming from breakpoint. 

## Usage ##

### Dataset ###

Two example datasets are provided:

  * toy_copy: A small toy autoencoding dataset from [TF Seq2seq toolkit](https://github.com/google/seq2seq/tree/2500c26add91b079ca00cf1f091db5a99ddab9ae).
  * iwslt14: The benchmark [IWSLT2014](https://sites.google.com/site/iwsltevaluation2014/home) (de-en) machine translation dataset. 

Download the data with the following cmds:

```
python prepare_data.py --data toy_copy
python prepare_data.py --data iwslt14
```

### Train the model ###

Train the model with the following cmd:

```
python seq2seq_attn_pg.py --config_model config_model --config_data config_toy_copy
```

Here:
  * `--config_model` specifies the model config. Note not to include the `.py` suffix.
  * `--config_data` specifies the data config.

All configs are (mostly) the same as those in the [seq2seq_attn example](../seq2seq_attn).

## Results ##

The code is for demonstrating Texar API. With pure policy gradient and without MLE pretraining the model collapse very quickly. 
