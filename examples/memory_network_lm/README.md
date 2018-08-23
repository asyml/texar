# End-to-End Memory Network for Language Modeling #

This example builds a Memory Network language model, and trains on PTB data. Model and training are described in   
[(Sukhbaatar, et. al.) End-To-End Memory Networks](https://arxiv.org/pdf/1503.08895v4.pdf). Model details are implemented in `texar.modules.memnet`.

Though the example is for language modeling, it is easy to adapt to other tasks like Question Answering, etc, as described in the above paper.

## Dataset ##

The standard [Penn Treebank (PTB) dataset](http://www.fit.vutbr.cz/~imikolov/rnnlm/) is used. 

If data does not exist under `data_path`, the program will automatically download the data. 

## Usage ##

The following cmd trains the model:

```bash
python3 lm_ptb_memnet.py --config config --data_path ./
```

Here:
  * `--config` specifies the config file to use. E.g., the above use the configuration defined in [config.py](./config.py).
  * `--data_path` specifies the directory containing PTB raw data (e.g., `ptb.train.txt`). If the data files do not exist, the program will automatically download, extract, and pre-process the data.
  * `--lr` specifies the initial learning rate. If not specified, the program will use the learning rate in the config file.

The model will begin training, and will evaluate on the validation data periodically, and evaluate on the test data after the training is done. Checkpoints are saved every 5 epochs.

## Configurations ##

[config.py](./config.py) is the largest and best configuration described on the last line of Table 2 in [(Sukhbaatar, et. al.) End-To-End Memory Networks](https://arxiv.org/pdf/1503.08895v4.pdf). It sets number of hops to 7, hidden dim to 150, and memory size to 200. This model has 4,582,500 parameters in total.

## Results ##

The perplexity of different configs is:

| config        | epochs | train | valid  | test  |
| ------------- | -------| ------| -------| ------|
| config        | 51     | 50.70 | 120.97 | 113.06|

This result of `config.py` is slightly inferior to the result presented in the paper, since the result in the paper is the best among 10 runs.
