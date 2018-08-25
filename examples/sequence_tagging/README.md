# Sequence tagging on CoNLL-2003 #

This example builds a bi-directional LSTM-CNN model for NER task and trains on CoNLL-2003 data. Model and training are described in   
>[(Ma et al.) End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](http://www.cs.cmu.edu/~xuezhem/publications/P16-1101.pdf)

The top CRF layer is not used here.

## Dataset ##

The code uses [CoNLL-2003 NER dataset](https://www.clips.uantwerpen.be/conll2003/ner/) (English). Please put data files (e.g., `eng.train.bio.conll`) under `./data` folder. Pretrained Glove word embeddings can also be used (set `load_glove=True` in [config.py](./config.py)). The Glove file should also be under `./data`. 

## Run ##

To train a NER model,

    python ner.py

The model will begin training, and will evaluate on the validation data periodically, and evaluate on the test data after the training is done. 

## Results ##

The results on validation and test data is:

|       |   prec   |  recall  |    F1    |
|-------|----------|----------|----------|
| valid |  91.18   |  92.41   |  91.79   |
| test  |  86.13   |  88.31   |  87.21   |

