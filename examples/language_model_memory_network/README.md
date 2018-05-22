# Language Model on PTB #

This example builds an Memory Network language model, and trains on PTB data. Model and training are described in   
[(Sukhbaatar, et. al.) End-To-End Memory Networks](https://arxiv.org/pdf/1503.08895v4.pdf). Model details are implemented in `texar.modules.memnet`.

The example shows:
  * Usage of `texar.modules.memnet` module.
  * Use of Texar with external Python data pipeline ([ptb_reader.py](./ptb_reader.py)).
  * Specification of various features of train op, like *gradient clipping*.

## Usage ##

The following cmd trains the model:

```
python lm_ptb_memnet.py --config config --data_path ./
```

Here:
  * `--config` specifies the config file to use. E.g., the above use the configuration defined in [config.py](./config.py), which is the largest config in the original paper.
  * `--data_path` specifies the directory containing PTB raw data (e.g., `ptb.train.txt`). If the data files do not exist, the program will automatically download, extract, and pre-process the data.

The model will begin training, and will evaluate on the validation data periodically, and evaluate on the test data after the training is done. Checkpoints are saved every 5 epochs.

## Results ##

The perplexity of different configs is:

| config        | epochs | train | valid  | test  |
| ------------- | -------| ------| -------| ------|
| config        | 51     | 50.70 | 120.97 | 113.06|
| config_memnet | 61     | 61.46 |  98.51 |  94.82|

Notice that config_memnet uses different learning rate decay scheme. Please check the commented part in `lm_pdb_memnet.py` and manually switch the scheme.
