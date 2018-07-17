# Sentence Sentiment Classifier #

This example builds sentence convolutional classifier, and trains on [SST data](https://nlp.stanford.edu/sentiment/index.html). The example config [config_kim.py](./config_kim.py) corresponds to the paper 
[(Kim) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf). 

The example shows:
  * Contruction of simple model, involving the `Embedder` and `Conv1DClassifier`.
  * Use of Texar `MultiAlignedData` to read parallel text and label data. 

## Usage ##

Use the following cmd to download and prepare the SST binary data:

```
python sst_data_preprocessor.py [--data_path ./data]
```

Here
  * `--data_path` specifies the directory to store the SST data. If the data files do not exist, the program will automatically download, extract, and pre-process the data.

The following cmd trains the model with Kim's config:

```
python clas_main.py --config config_kim
```

Here:
  * `--config` specifies the config file to use. E.g., the above use the configuration defined in [config_kim.py](./config_kim.py)

The model will begin training and evaluating on the validation data, and will evaluate on the test data after every epoch if a valid accuracy is obtained. 

## Results ##

The model achieves around `83%` test set accuracy.
