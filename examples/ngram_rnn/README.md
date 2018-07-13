# NGram-RNN #

This example provides the ngram-rnn model descrived in ..., and some other advanced algorithms on sequence to sequence model, including RAML(described in [Reward Augmented Maximum Likelihood for Neural Structured Prediction](https://arxiv.org/abs/1609.00150)) and Scheduled Sampling(described in [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://arxiv.org/abs/1506.03099)).

We also provide the data of 2 tasks, machine translation(iwslt 14) and text summarization(gigaword), as described in our paper.

## Usage ##

### Dataset ###

Our dataset can be found [here](https://drive.google.com/open?id=1-3QJyFZt68mrZoFrwlU8MMr7DuhvVrRd), decompress it and you can do the training following the commands below.



### NGram-RNN ###

The following command trains a ngram-rnn model:

```
python ngram-rnn.py --task=mt --k=3 --c=0.2 --d=0.1 
```

Here:
  * `--task` specifies the task, `mt` for machine translation and `ts` for text summarization.
  * `--k`  specifies the reinforce length described in the paper.
  * `--c, --d, --e` the coefficient to the reinforce items in the loss function, affiliated to the following 3 words separately.

The model will train the whole training set once, and evaluate on the validation data and test data periodically.



### RAML

The following command trains a raml model. Before using it you need a augmented file in the specified format, and we have provided in the data packet:

```
python raml.py --task=ts --raml_file=data/giga/samples_rouge.txt --n_samples=5
```

Here:

- `--task` specifies the task, `mt` for machine translation and `ts` for text summarization.
- `--raml_file`  the path to the raml file.
- `--n_samples` specifies the number of samples in the augmentation.

The model will train the whole training set once, and evaluate on the validation data and test data periodically.



### Scheduled Sampling

The following command trains a scheduled sampling model:

```
python scheduled_sampling.py --task=mt --decay_k=3000.
```

Here:

- `--task` specifies the task, `mt` for machine translation and `ts` for text summarization.
- `--decay_k`  specifies the parameter in the inverse sigmoid function controlling the increasing of sampling probability.

The model will train the whole training set once, and evaluate on the validation data and test data periodically.

