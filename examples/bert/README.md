# BERT: Pre-trained models and downstream applications

This is a Texar implementation of Google's BERT model, which allows to load pre-trained model parameters downloaded from the [official releaes](https://github.com/google-research/bert) and build/fine-tune arbitrary downstream applications with **distributed training** (This example showcases BERT for sentence classification).

With Texar, building the BERT model is as simple as creating a [`TransformerEncoder`](https://texar.readthedocs.io/en/latest/code/modules.html#transformerencoder) instance. We can initialize the parameters of the TransformerEncoder using a pre-trained BERT checkpoint by calling `init_bert_checkpoint(path_to_bert_checkpoint)`. 

In sum, this example showcases:

* Use of pre-trained Google BERT models in Texar
* Building and fine-tuning on downstream tasks
* Distributed training of the models

## Quick Start

### Download Dataset

We explain the use of the example code based on the Microsoft Research Paraphrase Corpus (MRPC) corpus for sentence classification. 

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

For **single-GPU** training (and evaluation), run the following cmd. The training updates the classification layer and fine-tunes the pre-trained BERT parameters.
```
    python bert_classifier_main.py --do_train --do_eval
    [--task=mrpc]
    [--config_bert_pretrain=uncased_L-12_H-768_A-12]
    [--config_downstream=config_classifier]
    [--config_data=config_data_mrpc]
    [--output_dir=output] 
```
Here:

- `task`: Specifies which dataset to experiment on.
- `config_bert_pretrain`: Specifies the architecture of pre-trained BERT model to use.
- `config_downstream`: Configuration of the downstream part. In this example, [`config_classifier.py`](https://github.com/asyml/texar/blob/master/examples/bert/bert_classifier_main.py) configs the classification layer and the optimization method.
- `config_data`: The data configuration.
- `output_dir`: The output path where checkpoints and summaries for tensorboard visualization are saved.

For **Multi-GPU training** on one or multiple machines, you may first install the prerequisite OpenMPI and Hovorod packages, as detailed in the [distributed_gpu](https://github.com/asyml/texar/tree/master/examples/distributed_gpu) example. 

Then run the following cmd for training and evaluation. The cmd trains the model on local with 2 GPUs. Evaluation is performed with the single rank-0 GPU.
```
mpirun -np 2 \
    -H  localhost:2\
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl tcp,self \
    -mca btl_tcp_if_include ens3 \
    python bert_classifier_main.py --do_train --do_eval --distributed
    [--task=mrpc]
    [--config_bert_pretrain=uncased_L-12_H-768_A-12]
    [--config_downstream=config_classifier]
    [--config_data=config_data_mrpc]
    [--output_dir=output] 
```
The key configurations of multi-gpu training:

* `-np`: total number of processes
* `-H`: IP addresses of different servers and the number of processes used in each server. For example, `-H 192.168.11.22:1,192.168.33.44:1`

Please refer to [distributed_gpu](https://github.com/asyml/texar/tree/master/examples/distributed_gpu) example for more details of the other multi-gpu configurations.

Note that we also specified the `--distributed` flag for multi-gpu training.

&nbsp;

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
