# BERT: Pre-trained models and downstream applications

This is a Texar implementation of Google's BERT model, which allows to load pre-trained model parameters downloaded from the [official release](https://github.com/google-research/bert) and build/fine-tune arbitrary downstream applications with **distributed training** (This example showcases BERT for sentence classification).

Texar provides ready-to-use modules including
[`BERTEncoder`](https://texar.readthedocs.io/en/latest/code/modules.html#bertencoder),
and [`BERTClassifier`](https://texar.readthedocs.io/en/latest/code/modules.html#bertclassifier).
This example shows the use of `BERTClassifier` for sentence classification tasks.

In sum, this example showcases:

* Use of pre-trained Google BERT models in Texar
* Building and fine-tuning on downstream tasks
* Distributed training of the models
* Use of Texar `TFRecordData` module for data loading and processing

## Quick Start

### Download Dataset

We explain the use of the example code based on the Microsoft Research Paraphrase Corpus (MRPC) corpus for sentence classification.

Download the data with the following command:

```
python data/download_glue_data.py --tasks=MRPC
```

By default, it will download the MRPC dataset into the `data` directory. FYI, the MRPC dataset is part of the [GLUE](https://gluebenchmark.com/tasks) dataset collection.

### Prepare data

We first preprocess the downloaded raw data into [TFRecord](https://www.tensorflow.org/tutorials/load_data/tf_records) files. The preprocessing tokenizes raw text with BPE encoding, truncates sequences, adds special tokens, etc.
Run the following command to this end: 

```
    python prepare_data.py --task=MRPC
    [--max_seq_length=128]
    [--pretrained_model_name=bert-base-uncased]
    [--tfrecord_output_dir=data/MRPC] 
```

- `--task`: Specifies the dataset name to preprocess. BERT provides default support for `{'CoLA', 'MNLI', 'MRPC', 'XNLI', 'SST'}` data.
- `--max_seq_length`: The maxium length of sequence. This includes BERT special tokens that will be automatically added. Longer sequence will be trimmed. 
- `--pretrained_model_name`: The name of pre-trained BERT model. See the [doc](https://texar.readthedocs.io/en/latest/code/modules.html#texar.tf.modules.PretrainedBERTMixin) for all supported models.
- `--tfrecord_output_dir`: The output path where the resulting TFRecord files will be put in. Be default, it is set to `data/{task}` where `{task}` is the (upper-cased) dataset name specified in `--task` above. So in the above cmd, the TFRecord files are output to `data/MRPC`.

**Outcome of the Preprocessing**:

- The preprocessing will output 3 TFRecord data files `{train.tf_record, eval.tf_record, test.tf_record}` in the specified output directory.

- The command also prints logs as follows:

  ```
    INFO:tensorflow:Loading data
    INFO:tensorflow:num_classes:2; num_train_data:3668
    INFO:tensorflow:config_data.py has been updated
    INFO:tensorflow:Data preparation finished
  ```
  **Note that** the data info `num_classes` and `num_train_data`, as well as `max_seq_length` specified in the cmd, are required for BERT training in the following. They should be specified in the data configuration file passed to BERT training (see below). 
- For convenience, the above cmd automatically writes `num_classes`, `num_train_data` and `max_seq_length` to `config_data.py`.

### Train and Evaluate

For **single-GPU** training (and evaluation), run the following cmd. The training updates the classification layer and fine-tunes the pre-trained BERT parameters.

```
    python bert_classifier_main.py --do_train --do_eval
    [--config_downstream=config_classifier]
    [--config_data=config_data]
    [--output_dir=output]
```
Here:

- `config_downstream`: Configuration of the downstream part. In this example, [`config_classifier`](./config_classifier.py) configures the classification layer and the optimization method.
- `config_data`: The data configuration. See the default [`config_data.py`](./config_data.py) for example. Make sure to specify `num_classes`, `num_train_data`, `max_seq_length`, and `tfrecord_data_dir` as used or output in the above [data preparation](#prepare-data) step.
- `output_dir`: The output path where checkpoints and TensorBoard summaries are saved.
- `pretrained_model_name`: The name of pre-trained BERT model. See the [doc](https://texar.readthedocs.io/en/latest/code/modules.html#texar.tf.modules.PretrainedBERTMixin) for all supported models.


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
    [--config_downstream=config_classifier]
    [--config_data=config_data]
    [--output_dir=output] 
```
The key configurations of multi-gpu training:

* `-np`: total number of processes
* `-H`: IP addresses of different servers and the number of processes used in each server. For example, `-H 192.168.11.22:1,192.168.33.44:1`

Please refer to [distributed_gpu](https://github.com/asyml/texar/tree/master/examples/distributed_gpu) example for more details of the other multi-gpu configurations.

Make sure to specifiy the `--distributed` flag as above for multi-gpu training.

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

`bert_classifier_main.py` also support other datasets/tasks. To do this, specify a different value to the `--task` flag when running [data preparation](#prepare-data).

For example, use the following commands to download the SST (Stanford Sentiment Treebank) dataset and run for sentence classification. Make sure to specify the correct data path and other info in the data configuration file.

```
python data/download_glue_data.py --tasks=SST
python prepare_data.py --task=SST
python bert_classifier_main.py --do_train --do_eval --config_data=config_data
```
