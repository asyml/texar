# Seq2seq Model #

This example builds an seq2seq model.

## Usage ##

### Dataset ###

Our dataset can be found [here](https://drive.google.com/open?id=1muBmgzV6Hm2MZbABKpkog71ZEoSO2c44), decompress it and you can do the training following the commands below.

### Training & Evaluate ###

```
python seq2seq.py --num_epochs=12
```

Here:
  * `--num_epochs` specifies the number of epochs(10 by default), every epoch the model will train the whole training set once, and then evaluate on the validation data and test data separately.



