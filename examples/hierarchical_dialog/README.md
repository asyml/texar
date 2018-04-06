# Hierarchical Recurrent Encoder-Decoder (HRED) Dialogue Model

This example builds a HRED dialogue model, training with [switchboard-1 Release 2](https://catalog.ldc.upenn.edu/ldc97s62). This example follows the setting of baseline from [(Zhao et al. ) Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders](https://arxiv.org/abs/1703.10960).

The basic HRED structure is described in [(Serban et al.) Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models](https://arxiv.org/abs/1507.04808).

This example shows:
+ Construct data for seq-to-seq dialogue model.
+ Specification of RNN cell config, like num\_units, dropout and orthogonal initialization.
+ Seq-to-seq model and MLP connector.
+ Decode with greedy inference and beam search.
+ Usage of beam search.

## Usage

To train the model, run

```
python hierarchical_dialog.py --stage train valid [--save_root <dir_to_save>]
```

If `save_root` is not specifided, model will be saved into `/tmp`, with name of
`hierarchical_example_best.ckpt`. The training process will stop when the valid loss doesn't decrease for 5 epochs.

To test the model, run

```
python hierarchical_dialog.py --load_path <path_to_save> --stage test [--test_batch_num <int>]
```

It will output the `bleu_recall` and `bleu_precision` following the definition in Zhao's paper, and a file `test_txt_results.txt` illustrate the predicted responses captured by beam search (with width 5).

You can use `test_batch_num` to limit the number of test samples.




