# BERT: Pre-trained models and downstream applications

This is a Texar implementation of Google's BERT model, which allows to load pre-trained model parameters downloaded from the [official releaes](https://github.com/google-research/bert) and build/fine-tune arbitrary downstream applications (This example showcases BERT for sentence classification).

With Texar, building the BERT model is as simple as constructing a [`TransformerEncoder`](https://texar.readthedocs.io/en/latest/code/modules.html#transformerencoder) instance. We can initialize the parameters of the TransformerEncoder using a pre-trained BERT checkpoint by calling `_init_bert_checkpoint(path_to_bert_checkpoint)`. 

## Quick Start

### Download Dataset

This example uses the Microsoft Research Paraphrase Corpus (MRPC) corpus for sentence classification, same as in the [BERT official release](https://github.com/google-research/bert#sentence-and-sentence-pair-classification-tasks). 

Download the data by
```
python download_glue_data.py --tasks=MRPC
```
By default, it will download the MRPC dataset into the `data` directory. FYI, the MRPC dataset part of the [GLUE](https://gluebenchmark.com/tasks) dataset collection.

### Download BERT Pre-train Model

```
sh download_model.sh
```
By default, it will download the `uncased_L-12_H-768_A-12.zip` and unzip it the same directory.
In the `bert_released_models/uncased_L-12_H-768_A-12.zip` directory, you may find five files.
- `bert-config.json` Model configurations for the BERT. Generally, it's a uncased-vocabulary, 12-layer, 768-hidden, 12-heads Transformer model, even it there is some trivial variant compared to the official Transformer.

### Train and Evaluate
To train the classifier and evaluate on the dev set, run the following cmd. The training updates the classification layer and fine-tunes the pre-trained BERT parameters.
```
python example_classifier.py --do_train --do_eval
[--config_bert_pretrain=uncased_L-12_H-768_A-12]
[--config_downstream=config_classifier]
[--config_data=config_data_mrpc]
[--output_dir=output] 
```
- `config_bert_pretrain`: Specifies the architecture of pre-trained BERT model to use.
- `config_downstream`: Configuration of the downstream part. In this example, [`config_classifier.py`](https://github.com/haoransh/texar_private/blob/master/examples/bert/config_classifier.py) configs the classification layer and the optimization method.
- `config_data`: The data configuration.
- `output_dir`: The output path where checkpoints and summaries for tensorboard visualization are saved.

After convergence, the evaluation performance is around the following. Due to certain randomness (e.g., random initialization of the classification layer), the evaluation accuracy is reasonable as long as it's `>0.84`.
```
INFO:tensorflow:evaluation loss:0.39845473161332456 accuracy:0.8848039215686274 eval_size:408
```

### Restore and Test
```
python example_classifier.py --do_test --checkpoint=output/model.ckpt
```
The output is by default saved in `output/test_results.tsv`, where each line contains output for each sample, with two fields representing the probabilities for each class.


## Hands-on tutorial

For detailed tutorial of using your own data and constructing your own model based on BERT, please refer to the notebook [`hands_on_tutorial_for_customed_dataset.ipynb`](https://github.com/haoransh/texar_private/blob/master/examples/bert/hands_on_tutorial_for_customed_dataset.ipynb)
