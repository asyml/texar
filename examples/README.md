# Examples #

Rich examples are included to demonstrate the use of Texar. The implementations of cutting-edge models/algorithms also provide references for reproducibility and comparisons. 

More examples are continuously added...

## Examples by Models/Algorithms ##

### RNN / Seq2seq ###

* [language_model_ptb](./language_model_ptb): Basic RNN language model
* [distributed_gpu](./distributed_gpu): Basic RNN language model with distributed training
* [seq2seq_attn](./seq2seq_attn): Attentional seq2seq
* [seq2seq_configs](./seq2seq_configs): Seq2seq implemented with Texar model template
* [seq2seq_rl](./seq2seq_rl): Attentional seq2seq trained with policy gradient
* [seq2seq_exposure_bias](./seq2seq_exposure_bias): Various algorithms tackling exposure bias in sequence generation
* [hierarchical_dialog](./hierarchical_dialog): Hierarchical recurrent encoder-decoder model for conversation response generation
* [torchtext](./torchtext): Use of torchtext data loader

### Transformer (Self-attention) ###

* [transformer](./transformer): Transformer for machine translation
* [bert](./bert): Pre-trained BERT model for text representation
* [gpt-2](./gpt-2): Pre-trained OpenAI GPT-2 language model
* [vae_text](./vae_text): VAE with a transformer decoder for improved language modeling 

### Variational Autoencoder (VAE) ###

* [vae_text](./vae_text): VAE language model

### GANs / Discriminiator-supervision ###

* [seqGAN](./seqgan): GANs for text generation
* [text_style_transfer](./text_style_transfer): Discriminator supervision for controlled text generation

### Reinforcement Learning ###

* [seq2seq_rl](./seq2seq_rl): Attentional seq2seq trained with policy gradient.
* [seqGAN](./seqgan): Policy gradient for sequence generation
* [rl_gym](./rl_gym): Various RL algoritms for games on OpenAI Gym

### Memory Network ###

* [memory_network_lm](./memory_network_lm): End-to-end memory network for language modeling

### Classifier / Sequence Prediction ###  

* [bert](./bert): Pre-trained BERT model for text representation
* [sentence_classifier](./sentence_classifier): Basic CNN-based sentence classifier
* [sequence_tagging](./sequence_tagging): BiLSTM-CNN model for Named Entity Recognition (NER)

### Reward Augmented Maximum Likelihood (RAML) ###

* [seq2seq_exposure_bias](./seq2seq_exposure_bias): RAML and other learning algorithms for sequence generation 

---

## Examples by Tasks

### Language Modeling ###

* [gpt-2](./gpt-2): Pre-trained OpenAI GPT-2 language model
* [language_model_ptb](./language_model_ptb): Basic RNN language model
* [vae_text](./vae_text): VAE language model
* [seqGAN](./seqgan): GAN + policy gradient
* [memory_network_lm](./memory_network_lm): End-to-end memory network for language modeling

### Machine Translation ###

* [seq2seq_attn](./seq2seq_attn): Attentional seq2seq
* [seq2seq_configs](./seq2seq_configs): Seq2seq implemented with Texar model template.
* [seq2seq_rl](./seq2seq_rl): Attentional seq2seq trained with policy gradient.
* [seq2seq_exposure_bias](./seq2seq_exposure_bias): Various algorithms tackling exposure bias in sequence generation (MT and summarization as examples).
* [transformer](./transformer): Transformer for machine translation

### Dialog ###

* [hierarchical_dialog](./hierarchical_dialog): Hierarchical recurrent encoder-decoder model for conversation response generation.

### Text Summarization ###

* [seq2seq_exposure_bias](./seq2seq_exposure_bias): Various algorithms tackling exposure bias in sequence generation (MT and summarization as examples).

### Text Style Transfer ###

* [text_style_transfer](./text_style_transfer): Discriminator supervision for controlled text generation

### Classification ###

* [bert](./bert): Pre-trained BERT model for text representation
* [sentence_classifier](./sentence_classifier): Basic CNN-based sentence classifier

### Sequence Tagging ###

* [sequence_tagging](./sequence_tagging): BiLSTM-CNN model for Named Entity Recognition (NER)

### Games ###

* [rl_gym](./rl_gym): Various RL algoritms for games on OpenAI Gym

---

## MISC ##

### Distributed training ###

* [distributed_gpu](./distributed_gpu): Basic example of distributed training.
* [bert](./bert): Distributed training of BERT.

