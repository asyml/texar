# BERT: Pre-trained models and downstream applications

This is a Texar implementation of Google's BERT model, which allows to load pre-trained model parameters downloaded from the [official releaes](https://github.com/google-research/bert) and build/fine-tune arbitrary downstream applications (This example showcases BERT for sentence classification).

With Texar, building the BERT model is as simple as constructing a [`TransformerEncoder`](https://texar.readthedocs.io/en/latest/code/modules.html#transformerencoder) instance. We can initialize the parameters of the TransformerEncoder using a pre-trained BERT checkpoint by calling `_init_bert_checkpoint(path_to_bert_checkpoint)`. 

## Quick Start

### Dataset Download

This example uses the Microsoft Research Paraphrase Corpus (MRPC) corpus for sentence classification, same as in the [BERT official release](https://github.com/google-research/bert#sentence-and-sentence-pair-classification-tasks). 

Download the data by
```
python data/download_glue_data.py [--data_dir] [--tasks] [--path_to_mrpc]
```
By default, it will download the [GLUE](https://gluebenchmark.com/tasks) datasets into the `data` directory. The MRPC dataset for classification is included as part of GLUE.

### BERT Pretrain Model Download

```
cd bert_released_models
sh download_model.sh
cd ..
```
By default, it will download the `uncased_L-12_H-768_A-12.zip` and unzip it the same directory.
In the `bert_released_models/uncased_L-12_H-768_A-12.zip` directory, you may find five files.
- `bert-config.json` Model configurations for the BERT. Generally, it's a uncased-vocabulary, 12-layer, 768-hidden, 12-heads Transformer model, even it there is some trivial variant compared to the official Transformer.

### Train and Evaluate
```
python example_classifier.py --do_train --do_eval
[--bert_pretrain_config=uncased_L-12_H-768_A-12]
[--bert_config_format=texar]
[--config_model=config_classifier] [--config_data=config_data_mrpc]
[--output_dir=output] 
```
- `bert_pretrain_config`: specify the pretrained BERT model architecture to be used
-`bert_config_format`: The configuration format. Choose `json` if loaded from the config attached from the downloaded BERT model directory. Choose `texar` to load the customed writen configuration file for texar, which is stored in `bert_config_lib/config_model_[--bert_pretrain_config].py`.
- `config_model`: The downstream classification model configuration is set in `config_classifier.py` 
- `config_data`: The data configuration is set in `config_data_mrpc.py`.
- `output_dir': `The checkpoint and summary data for tensorboard visualization will be saved in `output_dir` directory, which is `./output` by default.

You can achieve the evaluation performance shown as follows.
```
INFO:tensorflow:evaluation loss:0.39845473161332456 accuracy:0.8848039215686274 eval_size:408
```

### Restore and Test
```
python example_classifier.py --do_test --saved_model=output/model.ckpt
```
The output is by default saved in `output/test_results.tsv`.
Each line will contain output for each sample, with two fields representing the probabilities for each class.


## Hands On tutorial

For more detailed tutorial on how to use pretrained BERT model in your own dataset and task, please refer to the notebook tutorial `hands_on_tutorial_for_customed_dataset.ipynb`
