# Hierarchical Recurrent Encoder-Decoder (HRED) Dialogue Model

This example builds a HRED dialogue model, training with [switchboard-1 Release 2](https://catalog.ldc.upenn.edu/ldc97s62). This example follows the setting of baseline from [(Zhao et al. ) Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders](https://arxiv.org/abs/1703.10960).

The basic HRED structure is described in [(Serban et al.) Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models](https://arxiv.org/abs/1507.04808).

This example shows:
+ Construct data for seq-to-seq dialogue model with extra meta.
+ Seq-to-seq model and MLP connector.

## Usage

To train the model, run

```
python hred.py
```

where evaluation will be done after each epoch. 

## Results

The form shows results of perplexity and BLEU-1 after 10 epochs

| perplexity | BLEU-1 recall | BLEU-1 prec |
| -----------| --------------| ------------|
| 22.73      | 0.414         | 0.376       |



