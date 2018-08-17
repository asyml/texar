# Sequence tagging on CoNLL-2003 #

This example builds a bi-directional LSTM-CNN model for NER task and trains on CoNLL-2003 data. Model and training are described in   
>[End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](http://www.cs.cmu.edu/~xuezhem/publications/P16-1101.pdf)

The example shows:
  * Contruction of simple model, involving the `Embedder` and `RNN Encoder`.
  * Use of Texar with external Python data pipeline ([conll_reader.py](./conll_reader.py)).
  * Specification of various features of train op, like *gradient clipping* and *lr decay*.

## Usage ##

first make the tmp directory:

    mkdir tmp

To train a NER model,

    python ner.py

The model will begin training, and will evaluate on the validation data periodically, and evaluate on the test data after the training is done. 

## Results ##

The results on validation and test data is:

|       |   prec   |  recall  |    F1    |
|-------|----------|----------|----------|
| valid |  91.18   |  92.41   |  91.79   |
| test  |  86.13   |  88.31   |  87.21   |

