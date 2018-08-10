# Text Style Transfer #

This example implements a simplified variant of the model from 

[Toward Controlled Generation of Text](https://arxiv.org/pdf/1703.00955.pdf)
Zhiting Hu, Zichao Yang, Xiaodan Liang, Ruslan Salakhutdinov, Eric Xing; ICML 2017

The model roughly has an architecture of `Encoder--Decoder--Classifier`. Compared to the paper, the following simplications are made:
  
  * Replaces the base Variational Autoencoder (VAE) model with an attentional Autoencoder (AE) -- VAE is not necessary in the text style transfer setting since we do not need to interpolate the latent space as in the paper.
  * Discriminator (i.e., attribute classifier) is pre-trained, and fixed throughout the full model training.
  * Independency constraint is omitted.

## Usage ##

### Dataset ###
Download the yelp sentiment dataset with the following cmd:

```
python prepare_data.py
```

### Train the model ###

Train the model on the above data to do sentiment transfer.

```
python main.py --config config
```

[config.py](./config.py) contains the data and mode configurations. 

* The model will first be pre-trained for a few epochs (specified in `config.py`). During pre-training, the `Encoder-Decoder` part is trained as an autoencoder, while the `Classifier` part is trained with the classification labels.
* Full-training is then performed for another few epochs. During full-training, the `Classifier` part is fixed, and the `Encoder-Decoder` part is trained to fit the classifier, along with continuing to minimize the autoencoding loss.

Training log is printed as below:
```
gamma: 1.0, lambda_g: 0.0
step: 1, loss_d: 0.6903 accu_d: 0.5625
step: 1, loss_g_clas: 0.6991 loss_g: 9.1452 accu_g: 0.2812 loss_g_ae: 9.1452 accu_g_gdy: 0.2969
step: 500, loss_d: 0.0989 accu_d: 0.9688
step: 500, loss_g_clas: 0.2985 loss_g: 3.9696 accu_g: 0.8891 loss_g_ae: 3.9696 accu_g_gdy: 0.7734
...
step: 6500, loss_d: 0.0806 accu_d: 0.9703
step: 6500, loss_g_clas: 5.7137 loss_g: 0.2887 accu_g: 0.0844 loss_g_ae: 0.2887 accu_g_gdy: 0.0625
epoch: 1, loss_d: 0.0876 accu_d: 0.9719
epoch: 1, loss_g_clas: 6.7360 loss_g: 0.2195 accu_g: 0.0627 loss_g_ae: 0.2195 accu_g_gdy: 0.0642
val: accu_g: 0.0445 loss_g_ae: 0.1302 accu_d: 0.9774 bleu: 90.7896 loss_g: 0.1302 loss_d: 0.0666 loss_g_clas: 7.0310 accu_g_gdy: 0.0482
...

```
where:
- `loss_d` and `accu_d` are the classification loss/accuracy of the `Classifier` part.
- `loss_g_clas` is the classification loss of the generated sentences.
- `loss_g_ae` is the autoencoding loss.
- `loss_g` is the joint loss `= loss_g_ae + lambda_g * loss_g_clas`.
- `accu_g` is the classification accuracy of the generated sentences with soft represetations (i.e., Gumbel-softmax).
- `accu_g_gdy` is the classification accuracy of the generated sentences with greedy decoding.
- `bleu` is the BLEU score between the generated and input sentences.

## Results ##

Text style transfer has two primary goals:
1. The generated sentence should have desired attribute (e.g., positive/negative sentiment)
2. The generated sentence should keep the content of the original one

We use automatic metrics to evaluate both: 
* For (1), we can use a pre-trained classifier to classify the generated sentences and evaluate the accuracy (the higher the better). In this code we have not implemented a stand-alone classifier for evaluation, which could be very easy though. The `Classifier` part in the model gives a reasonably good estimation (i.e., `accu_g_gdy` in the above) of the accuracy. 
* For (2), we evaluate the BLEU score between the generated sentences and the original sentences, i.e., `bleu` in the above (the higher the better) (See [Yang et al., 2018](https://arxiv.org/pdf/1805.11749.pdf) for more details.)

The implementation here gives the following performance after 10 epochs of pre-training and 2 epochs of full-training:

| Accuracy (by the `Classifier` part)  | BLEU (with the original sentence) |
| -------------------------------------| ----------------------------------|
|  |  |

