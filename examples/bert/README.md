# BERT: Pre-trained models and downstream applications

This is a Texar implementation of Google's BERT model, which allows to load pre-trained model parameters downloaded from the [official releaes](https://github.com/google-research/bert) and build/fine-tune arbitrary downstream applications (This example showcases BERT for sentence classification).

With Texar, building the BERT model is as simple as creating a [`TransformerEncoder`](https://texar.readthedocs.io/en/latest/code/modules.html#transformerencoder) instance. We can initialize the parameters of the TransformerEncoder using a pre-trained BERT checkpoint by calling `init_bert_checkpoint(path_to_bert_checkpoint)`. 

## Quick Start

### Download Dataset

We explain the use of the example code based on the Microsoft Research Paraphrase Corpus (MRPC) corpus for sentence classification. See 

Download the data with the following cmd
```
python data/download_glue_data.py --tasks=MRPC
```
By default, it will download the MRPC dataset into the `data` directory. FYI, the MRPC dataset part of the [GLUE](https://gluebenchmark.com/tasks) dataset collection.

### Download BERT Pre-train Model

```
sh bert_pretrained_models/download_model.sh
```
By default, it will download a pretrained model (BERT-Base Uncased: 12-layer, 768-hidden, 12-heads, 110M parameters) named `uncased_L-12_H-768_A-12` to `bert_pretrained_models/`.

Under `bert_pretrained_models/uncased_L-12_H-768_A-12`, you can find 5 files, where
- `bert-config.json` is the model configuration of the BERT model. For the particular model we just downloaded, it is an uncased-vocabulary, 12-layer, 768-hidden, 12-heads Transformer model.

### Train and Evaluate

To train the classifier and evaluate on the dev set, run the following cmd. The training updates the classification layer and fine-tunes the pre-trained BERT parameters.
```
python bert_classifier_main.py --do_train --do_eval
[--task=mrpc]
[--config_bert_pretrain=uncased_L-12_H-768_A-12]
[--config_downstream=config_classifier]
[--config_data=config_data_mrpc]
[--output_dir=output] 
```
- `task`: Specifies which dataset to experiment on.
- `config_bert_pretrain`: Specifies the architecture of pre-trained BERT model to use.
- `config_downstream`: Configuration of the downstream part. In this example, [`config_classifier.py`](https://github.com/asyml/texar/blob/master/examples/bert/bert_classifier_main.py) configs the classification layer and the optimization method.
- `config_data`: The data configuration.
- `output_dir`: The output path where checkpoints and summaries for tensorboard visualization are saved.

After convergence, the evaluation performance is around the following. Due to certain randomness (e.g., random initialization of the classification layer), the evaluation accuracy is reasonable as long as it's `>0.84`.
```
INFO:tensorflow:dev accu: 0.8676470588235294
```

### Restore and Test

``
python bert_classifier_main.py --do_test --checkpoint=output/model.ckpt
``

The output is by default saved in `output/test_results.tsv`, where each line contains the predicted label for each sample.


## Use other datasets/tasks

`bert_classifier_main.py` also support other datasets/tasks. To do this, specify a different value to the `--task` flag, and use a corresponding data configuration file. 

For example, use the following commands to download the SST (Stanford Sentiment Treebank) dataset and run for sentence classification.
```
python data/download_glue_data.py --tasks=SST
python bert_classifier_main.py --do_train --do_eval --task=sst --config_data=config_data_sst
```
