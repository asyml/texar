# Seq2seq Models #

Attentional seq2seq models for machine translation, including
* Baseline: Plain Attentional seq2seq
* Scheduled Sampling: Described in [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://arxiv.org/abs/1506.03099)
* Reward Augmented Maximum Likelihood(RAML): Described in [
Softmax Q-Distribution Estimation for Structured Prediction: A Theoretical Interpretation for RAML](https://arxiv.org/abs/1705.07136)
* Interpolation Algorithm: Described in [Connecting the Dots Between MLE and RL for Sequence Generation](https://www.cs.cmu.edu/~zhitingh/)

## Usage ##

### Dataset ###

Two example datasets are provided:

  * iwslt14: The benchmark [IWSLT2014](https://sites.google.com/site/iwsltevaluation2014/home) (de-en) machine translation dataset, following [(Norouzi et al., 2016)]() for data pre-processing.
  * gigaword: The benchmark text summurization dataset(training set is 200,000 samples). 

Download the data with the following cmds:

```
python utils/prepare_data.py --data iwslt14
python utils/prepare_data.py --data giga
```

### Train the models ###

#### Baseline Attentional Seq2seq

```
python baseline_seq2seq_attn_main.py \
    --config_model configs.config_model \
    --config_data configs.config_iwslt14
```

Here:
  * `--config_model` specifies the model config. Note not to include the `.py` suffix.
  * `--config_data` specifies the data config.

[configs.config_model.py](./configs.config_model.py) specifies a single-layer seq2seq model with Luong attention and bi-directional RNN encoder. Hyperparameters taking default values can be omitted from the config file. 

For demonstration purpose, [configs.config_model_full.py](./configs.config_model_full.py) gives all possible hyperparameters for the model. The two config files will lead to the same model.

#### Reward Augmented Maximum Likelihood(RAML)
```
python raml_main.py \
    --config_model configs.config_model \
    --config_data configs.config_iwslt14 \
    --raml_file data/iwslt14/samples_iwslt14.txt \
    --n_samples 10
```
Here:
  * `--raml_file` specifies a file containin the augmented samples and rewards.
  * `--n_samples` specifies number of augmented samples for every target sentence.
  * `--tau` specifies the hyperparameter in RAML algorithm.

We have provided raml_file including augmented samples for both ```iwslt14``` and ```gigaword``` in the dataset you can download. If you need to tune some hyperparameters for RAML, we also provide the scripts for samples generating and you can refer to [utils/raml_samples_generation](utils/raml_samples_generation).


#### Scheduled Sampling
```
python scheduled_sampling_main.py \
    --config_model configs.config_model \
    --config_data configs.config_iwslt14 \
    --decay_factor 500.
```
Here:
  * `--decay_factor` specifies the hyperparameter controling the speed of increasing the probability of sampling from model.


#### Interpolation Algorithm
```
python interpolation_main.py \
    --config_model configs.config_model \
    --config_data configs.config_iwslt14 \
    --lambdas_init [0.04,0.06,0.0] \
    --delta_lambda_self 0.06 \
    --delta_lambda_reward 0.06 \
    --lambda_reward_steps 4
```
Here:

  * `--lambdas_init` specifies the initial value of lambdas.
  * `--delta_lambda_reward` specifies the increment of lambda_reward every annealing
  * `--delta_lambda_self` specifies the decrement of lambda_self every annealing.
  * `--k` specifires the times of increasing lambda_reward after incresing lambda_self once.

## Results ##

### Machine Translation
| Model      | BLEU Score   | 
| -----------| -------|  
| MLE        | 26.44 ± 0.18  | 
| Scheduled Sampling   | 26.76  ± 0.17  |
| RAML | 27.22  ± 0.14  |
| Interpolation | 27.82  ± 0.11  |

### Text Summarization
| Model      | Rouge-1   | Rouge-2 | Rouge-L |  
| -----------| -------|-------|-------|  
| MLE        | 36.11 ± 0.21  | 16.39 ± 0.16 | 32.32 ± 0.19 |
| Scheduled Sampling   |  36.59 ± 0.12  |16.79 ± 0.22|32.77 ± 0.17|
| RAML | 36.30  ± 0.04  | 16.69 ± 0.20 | 32.49 ± 0.17 |
| Interpolation | 36.72  ± 0.29  |16.99 ± 0.17 | 32.95 ± 0.33|

 
