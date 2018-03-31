# Language Model on PTB #

This example builds an LSTM language model, and trains on PTB data. 

The example shows:
  * Contruction of simple model, involving the Embedder and RNN Decoder.
  
  * Use of Texar with external Python data pipeline ([ptb_reader.py](https://github.com/ZhitingHu/txtgen/blob/master/examples/language_model_ptb/ptb_reader.py)).
  
  * Specification of various features of train op, like gradient clipping and lr decay.

## Setup and Run ##

