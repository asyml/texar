# Bert

This is an implementation of the Google's BERT implementation.

## Prerequisites
```
pip install -r requirements
```
## Quick trial

Here we give a simple example on how to initialize your model with BERT for sentence classification task, 
which is described in [BERT official release](https://github.com/google-research/bert#sentence-and-sentence-pair-classification-tasks).
We choose the experiments on Microsoft Research Paraphrase Corpus (MRPC) corpus as the showcase.

### Dataset Download
```
python download_glue_data.py [--data_dir] [--tasks] [--path_to_mrpc]
```
By default, it will download GLUE datasets into `glue_data` directory. For more information on GLUE, you can refer to 
[gluebenchmark](https://gluebenchmark.com/tasks)

### BERT Pretrain Model Download

```
cd bert_released_models
sh download_model.sh
```
By default, it will download the `uncased_L-12_H-768_A-12.zip` and unzip it the same directory.
In the `bert_released_models/uncased_L-12_H-768_A-12.zip` directory, you may find five files.
- `bert-config.json` Model configurations for the BERT. Generally, it's a uncased-vocabulary, 12-layer, 768-hidden, 12-heads Transformer model, even it there is some trivial variant compared to the official Transformer.

### Train and Evaluate
```
python example_classifier.py --do_train --do_eval --bert_config_format=texar [--output_dir]
```

The checkpoint and summary data for tensorboard visualization will be saved in `output_dir`
directory, which is `./output` by default.
```
INFO:tensorflow:evaluation loss:0.39845473161332456 accuracy:0.8848039215686274 eval_size:408
```

### Restore and Test
```
python example_classifier.py --do_test --saved_model=output/model.ckpt
```
The output is by default saved in `output/test_results.tsv`.
Each line will contain output for each sample, with two fields representing the probabilities for each class.
