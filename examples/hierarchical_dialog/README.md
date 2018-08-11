# Hierarchical Recurrent Encoder-Decoder (HRED) Dialogue Model

This example builds a HRED dialogue model described in [(Serban et al.) Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models](https://arxiv.org/abs/1507.04808). 

The dataset used here is provided by [(Zhao et al. ) Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders](https://arxiv.org/abs/1703.10960), which adapts [switchboard-1 Release 2](https://catalog.ldc.upenn.edu/ldc97s62). In particular, for evaluation purpose, multiple reference responses for each dialog context in the test set are collected through manual annotations. 

This example demonstrates:
* Use of `MultiAlignedData` to read parallel data with multiple fields, e.g., (source, target, meta, ...)
* Use of the `'variable_utterance'` hyperparameter in TextData to read dialog history data.
* Use of the `'embedding_init'` hyperparameter in TextData to read pre-trained word embedding as initialization. 
* Use of `HierarchicalRNNEncoder` to encode dialog history with utterance-level and word-level encoding.
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

The table shows results of perplexity and BLEU after 10 epochs, comparing the results of [Zhao et al.](https://arxiv.org/abs/1703.10960). BLEU score is computed with 5 sampling.

|               | biminor | uniminor | Zhao et al. |
| --------------| --------| ---------| -------|
| Perlexity     | 22.73   | 23.51    | 35.4   |
| BLEU-1 recall | 0.414   | 0.409    | 0.405  |
| BLEU-1 prec   | 0.376   | 0.368    | 0.336  |
| BLEU-2 recall | 0.328   | 0.317    | 0.300  |
| BLEU-2 prec   | 0.295   | 0.289    | 0.281  |
| BLEU-3 recall | 0.281   | 0.283    | 0.272  |
| BLEU-3 prec   | 0.256   | 0.251    | 0.254  |
| BLEU-4 recall | 0.228   | 0.223    | 0.226  |
| BLEU-4 prec   | 0.205   | 0.211    | 0.215  |
