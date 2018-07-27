# Seq2seq Model #

This example builds an seq2seq model.

## Usage ##

### Dataset ###

Our dataset can be found [here](https://drive.google.com/open?id=1-3QJyFZt68mrZoFrwlU8MMr7DuhvVrRd), decompress it and you can do the training following the commands below.

### Training & Evaluate ###

```
python seq2seq.py --dataset=iwslt14 --metric=bleu --num_epochs=10
```

Here:
  * `--dataset` specifies the dataset, `iwslt14`  for iwslt14 de-en and `giga` for gigaword. 
  * `--metric`  specifies the metric in evaluation, can be `belu` or `rouge`.
  * `--num_epochs` specifies the number of epochs(10 by default), every epoch the model will train the whole training set once, and then evaluate on the validation data and test data separately.



