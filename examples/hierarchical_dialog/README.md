# Hierarchical Recurrent Encoder-Decoder (HRED) Dialogue Model

This example builds a HRED dialogue model, training with [switchboard-1 Release 2](https://catalog.ldc.upenn.edu/ldc97s62). This example follows the setting of baseline from [(Zhao et al. ) Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders](https://arxiv.org/abs/1703.10960).

The basic HRED structure is described in [(Serban et al.) Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models](https://arxiv.org/abs/1507.04808).

This example shows:
+ Construct data for seq-to-seq dialogue model with speaker token (flags denote the speaker for each utterance) as extra meta.
+ The way to build HRED pipeline and to add extra meta using predefined interface.
+ Usage of BLEU evaluation.

## Usage

To train the model, run

```
python hred.py --config config_model_biminor
```

where evaluation will be done after each epoch. 

Here:
+ `--config_model` specifies the model config. Note not to include the `.py` suffix. 

This example provides two configs `biminor` and `uniminor`, in which a bidirectional/unidirectional RNN is used as the utterance-level encoder respectively, while other configurations remain the same. 

## Results

The form shows results of perplexity and BLEU after 10 epochs, comparing to the implementation of Zhao et al.

|               | biminor | uniminor | Zhao's |
| --------------| --------| ---------| -------|
| perlexity     | 22.73   | 23.51    | 35.4   |
| BLEU-1 recall | 0.414   | 0.409    | 0.405  |
| BLEU-1 prec   | 0.376   | 0.368    | 0.336  |
| BLEU-2 recall | 0.328   | 0.317    | 0.300  |
| BLEU-2 prec   | 0.295   | 0.289    | 0.281  |
| BLEU-3 recall | 0.281   | 0.283    | 0.272  |
| BLEU-3 prec   | 0.256   | 0.251    | 0.254  |
| BLEU-4 recall | 0.228   | 0.223    | 0.226  |
| BLEU-4 prec   | 0.205   | 0.211    | 0.215  |

in which BLEU score is computed with 5 sampling.

