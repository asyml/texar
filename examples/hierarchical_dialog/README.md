# Hierarchical Recurrent Encoder-Decoder (HRED) Dialogue Model

This example builds a HRED dialogue model described in [(Serban et al. 2016) Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models](https://arxiv.org/abs/1507.04808). 

The dataset used here is provided by [(Zhao et al. 2017) Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders](https://arxiv.org/abs/1703.10960), which adapts [switchboard-1 Release 2](https://catalog.ldc.upenn.edu/ldc97s62). In particular, for evaluation purpose, multiple reference responses for each dialog context in the test set are collected through manual annotations. 

This example demonstrates:
* Use of `MultiAlignedData` to read parallel data with multiple fields, e.g., (source, target, meta, ...)
* Use of the `'variable_utterance'` hyperparameter in TextData to read dialog history data.
* Use of the `'embedding_init'` hyperparameter in TextData to read pre-trained word embedding as initialization. 
* Use of `HierarchicalRNNEncoder` to encode dialog history with utterance-level and word-level encoding.
* Use of *beam search decoding* and *random sample decoding* at inference time. 
* Addition of speaker meta-data in the encoder-decoder model.

## Usage

### Dataset

Download and preprocess the data with the following cmd:
```
python sw_loader.py
```
* Train/dev/test sets contain 200K, 5K, 5K examples, respectively.
* Vocab size is 10,000.
* `./data/switchboard/embedding.txt` contains word embeddings extracted from [glove.twitter.27B.200d](https://nlp.stanford.edu/projects/glove). You can also directly use the original glove.twitter.27B.200d file, and the Texar TextData module will automatically extract relevant embeddings for the vocabulary. 

### Train the model

To train the model, run

```
python hred.py --config_data config_data --config_model config_model_biminor 
```
Evaluation will be performed after each epoch. 

Here:
* `--config_data` specifies the data configuration.
* `--config_model` specifies the model configuration. Note not to include the `.py` suffix. Two configs are provided:
  - [biminor.py](./config_model_biminor.py) uses a bi-directional RNN as the word-level (minor-level) encoder
  - [uniminor.py](./config_model_uniminor.py) uses a uni-directional RNN as the word-level (minor-level) encoder

Both configs use a uni-directional RNN for the utterance-level (major-level) encoder

## Results

The table shows results of perplexity and BLEU after 10 epochs, comparing the results of [(Zhao et al. 2017)](https://arxiv.org/abs/1703.10960) (See "Baseline" of Table.1 in the paper). Note that:
* We report results of random sample decoding, which performs slightly better than beam search decoding. 
* `num_samples` is the number of samples generated for each test instances (for computing precision and recall of BLEU). See sec.5.2 of the paper for the definition of the metrics.
* (Zhao et al. 2017) uses more meta data besides the speaker meta-data here.
* Results may vary a bit due to randomness.

|               | biminor<br>num_samples=10   | biminor<br>num_samples=5 | Zhao et al.<br>num_samples=5 |
| --------------| ---------------| --------------| --------------|
| Perlexity     | 23.79          | 24.26         | 35.4   |
| BLEU-1 recall | 0.478          | 0.386         | 0.405  |
| BLEU-1 prec   | 0.379          | 0.395         | 0.336  |
| BLEU-2 recall | 0.391          | 0.319         | 0.300  |
| BLEU-2 prec   | 0.310          | 0.324         | 0.281  |
| BLEU-3 recall | 0.330          | 0.270         | 0.272  |
| BLEU-3 prec   | 0.259          | 0.272         | 0.254  |
| BLEU-4 recall | 0.262          | 0.216         | 0.226  |
| BLEU-4 prec   | 0.204          | 0.215         | 0.215  |
