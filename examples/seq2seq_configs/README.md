# Seq2seq Model #

This example builds a (plain) seq2seq model with Texar's model template and Tensorflow estimator. 

## Usage ##

### Dataset ###

Download the example dataset:

  * toy_copy: A small toy autoencoding dataset from [TF Seq2seq toolkit](https://github.com/google/seq2seq/tree/2500c26add91b079ca00cf1f091db5a99ddab9ae).

```
python [PATH_TEXAR]/examples/seq2seq_attn/prepare_data.py --data toy_copy
```

### Train the model ###

Train the model with the following cmd:

```
python [PATH_TEXAR]/bin/train.py --config_paths config_model_small.yml,config_data_toy_copy.yml 
```

See [train.py](../../bin/train.py) for other available configurations.

[config_model_small.yml](./config_model_small.yml) speicifies a small-size model with single-layer RNN encoder/decoder. [config_model_medium.yml](./config_model_medium.yml) specifies a medium-size one with 2-layer RNN encoder/decoder.

The model will be trained/evaluated/checkpointed within the [Tensorflow Estimator](https://www.tensorflow.org/guide/estimators).
