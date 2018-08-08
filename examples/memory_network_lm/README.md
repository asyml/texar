# Language Model on PTB #

This example builds a Memory Network language model, and trains on PTB data. Model and training are described in   
[(Sukhbaatar, et. al.) End-To-End Memory Networks](https://arxiv.org/pdf/1503.08895v4.pdf). Model details are implemented in `texar.modules.memnet`.

The example shows:
  * Usage of `texar.modules.memnet` module.
  * Use of Texar with external Python data pipeline ([ptb_reader.py](./ptb_reader.py)).
  * Specification of various features of train op, like *gradient clipping*.

## Dataset ##

The code uses Penn Treebank (PTB) dataset, which is the same as the dataset used in the `language_model_ptb` example.

The data required for this example is in the `data/` dir of the PTB dataset from Tomas Mikolov's webpage:

```bash
wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar xvf simple-examples.tgz
```

If data is now provided, the program will automatically download from above into the current directory.

## Usage ##

The following cmd trains the model:

```bash
python3 lm_ptb_memnet.py --config config_memnet --data_path ./
```

Here:
  * `--config` specifies the config file to use. E.g., the above use the configuration defined in [config_memnet.py](./config_memnet.py).
  * `--data_path` specifies the directory containing PTB raw data (e.g., `ptb.train.txt`). If the data files do not exist, the program will automatically download, extract, and pre-process the data.

The model will begin training, and will evaluate on the validation data periodically, and evaluate on the test data after the training is done. Checkpoints are saved every 5 epochs.

## Configurations ##

There are two config files in this directory: [config.py](./config.py) and [config_memnet.py](./config_memnet.py).

[config.py](./config.py) is the largest and best configuration described on the last line of Table 2 in [(Sukhbaatar, et. al.) End-To-End Memory Networks](https://arxiv.org/pdf/1503.08895v4.pdf). It sets number of hops to 7, hidden dim to 150, and memory size to 200. This model has 4582500 parameters in total.

[config_memnet.py](./config_memnet.py) is a modified larger model which is used in comparison with other language models. It yields 11073600 parameters in total. It increases number of hops to 10, hidden dim to 360. In order to alleviate overfitting, dropout rate of 0.2 is added onto some parts of the model. Also, I referenced the idea of so called "variational" and share dropout masks among the outputs of different hops. Notice that this config uses a different learning rate decay scheme. See the config file and code for more details about this scheme.

## Results ##

The perplexity of different configs is:

| config        | epochs | train | valid  | test  |
| ------------- | -------| ------| -------| ------|
| config        | 51     | 50.70 | 120.97 | 113.06|
| config_memnet | 61     | 61.46 |  98.51 |  94.82|

This result of `config` is slightly inferior to the result presented in the paper, since the result in the paper is the best among 10 runs.
