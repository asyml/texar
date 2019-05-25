# GPT-2: Pre-trained Langauge Model

This is a Texar implementation of [OpenAI GPT-2 (Generative Pre-Trainning)](https://github.com/openai/gpt-2) language model, which allows to load official pre-trained model parameters, generate samples, and fine-tune the model, etc.

With Texar, building the GPT-2 model is as simple as creating a [`TransformerDecoder`](https://texar.readthedocs.io/en/latest/code/modules.html#transformerdecoder) instance. We can initialize the parameters of the TransformerDecoder using a pre-trained GPT-2 checkpoint by calling `init_gpt2_checkpoint(path_to_gpt2_checkpoint)` .

In sum, this example showcases:

* Contructing and using pre-trained GPT-2 models in Texar
* Using GPT-2 to generate text samples with or without context
* **Train or fine-tune** the model with **distributed GPU**
* Examples of other use cases

## Quick Start (I) - Generation with the Pre-trained Model

### Download GPT-2 Pre-trained Model

Download the GPT-2 `117M` model checkpoint with the following command:
```
sh gpt2_pretrained_models/download_model.sh model_117M
```
By default, it will download a pretrained model named `model_117M` to `gpt2_pretrained_models/`.

To download the GPT-2 `345M` model checkpoint, use:
```
sh gpt2_pretrained_models/download_model.sh model_345M
```

### Usage
| WARNING: Samples are unfiltered and may contain offensive content. |
| --- |

#### Interactive mode (to generate samples with context)

This mode will initialize an interactive interface, which allows users to type in the context sentence. The model then generates continuation of the context. Top-K sample decoding is used. By default, the GPT-2 `117M` model is used.

```
python gpt2_generate_main.py --is_interactive \
--max_decoding_length=100 \
--temperature=0.7 \
--top_k=40
```

Here:

- `is_interactive`: Specifies interactive mode.
- `max_decoding_length`: The maximum number of tokens in the sample. **Note that this includes tokens in the context**. 
- `temperature`: Softmax temperature of top-k sample decoding. Larger values (above 1.0) result in more random samples, while smaller values push the sampling distribution towards the argmax. Must be strictly greater than 0. Defaults to `0.7`.
- `top_k`: Number of top most likely candidates from a vocab distribution in each decoding step. Defaults to `40`.
- `nsamples`: Number of samples to generate for each input. 

To use the GPT-2 `345M` model, specify `--pretrain_checkpoint` and `--config_model`:

```
python gpt2_generate_main.py --is_interactive \
--max_decoding_length=100 \
--temperature=0.7 \
--top_k=40 \
--config_model=configs.config_model_345M \
--pretrain_checkpoint=gpt2_pretrained_models/model_345M/model.ckpt
--pretrain_model_dir=gpt2_pretrained_models/model_345M
```

Here:

- `pretrain_checkpoint`: Path to the model checkpoints. Default to `gpt2_pretrained_models/model_117M/model.ckpt`.
- `config_model`: Model configuration file. Default to `configs.config_model_117M`.
- `pretrain_model_dir`:  The directory of pretrained model, for loading vocabuary, etc. Default to `gpt2_pretrained_models/model_117M`. 

**Example input:**
```
Model input >>> Micheal Jordan is the greatest player in history !
```
**Example output:**
```
======================================== SAMPLE 1 ========================================

He's the one who has made all the difference. He's a true legend. He's a great athlete, 
a great athlete. He's a great athlete. I'm so happy for him. I'm so happy for his family, 
the family, and I'm so happy for him. I'm so happy for his teammates, his teammates, and 
I'm so happy for him.

The last time we saw him on stage, he

================================================================================
```

#### Non-interactive mode (to generate samples from scratch)

This mode generates a batch of samples from scratch.

```
python gpt2_generate_main.py
--nsamples=1 \
--batch_size=1 \
--max_decoding_len=100 \
--temperature=0.7 \
--top_k=40
```

Here:

- `nsamples`: Total number of samples to generate, must be dividable by the `batch_size`.
- `batch_size`: Each iteration generates `batch_size` number of samples.

To use GPT-2 `345M` model, specify `--pretrain_checkpoint`, `--config_model` and `--pretrain_model_dir` as above.

**Example output:**

```
"A new government and a healthy economy have a chance to take this up."

After he said the election's outcome in the House was important and had helped to build 
confidence in the House, former Ukip leader Nigel Farage spoke about working to boost 
the economy, saying the vote for the "lefties" and others "were bad optics for Labour 
in this way".
```

## Quick Start (II) - Fine-tune the Pre-trained Model 

This section shows how we can fine-tune the pre-trained GPT2 model and use the resulting model for generation.

First of all, **download** the pre-trained model [as above](https://github.com/asyml/texar/tree/master/examples/gpt-2#download-gpt-2-pre-trained-model).

### Prepare data

We first preprocess data with the GPT-2 BPE encoding. 

A toy dataset is provided under [`data/toy/`](data/toy) which includes `train.txt`, `dev.txt`, and `test.txt`. This example will fit the GPT-2 model on `train.txt`, evaluate perplexity on `dev.txt`, and do continuation generation using `test.txt` as the context.

Run the following cmd to transform the data into [TFRecord](https://www.tensorflow.org/tutorials/load_data/tf_records) format and perform processing such as truncation, BPE encoding, adding special tokens, etc:

```
    python prepare_data.py --data_dir data/toy 
    [--max_seq_length=128]
    [--tfrecord_output_dir=data/toy] 
    [--pretrain_model_dir=gpt2_pretrained_models/model_117M]
```
- `data_dir`: The directory of raw data, wherein data files must be named as 'train.txt', 'dev.txt', or 'test.txt'. It is *not* necessary to provide all three files.
- `max_seq_length`: The maxium length of sequence after BPE encoding. This includes GPT-2 special tokens that will be automatically added. Longer sequence will be trimmed. 
- `tfrecord_output_dir`: The output path where the resulting TFRecord files will be put in. Be default, it is set to be the same as `data_dir`. 
- `pretrain_model_dir`: The downloaded pretrained model directory, wherein the vocabulary files are used for data processing.

The above cmd will output TFRecord files in the specified output directory. E.g., if `train.txt` is provided under `data_dir`, the output file `train.tf_record` will be produced under `tfrecord_output_dir`. 

### Train and Evaluate

For **single-GPU** training (and evaluation), run the following cmd. The cmd fine-tunes the pre-trained GPT-2 parameters, and evalautes perplexity on the dev set.
```
    python gpt2_train_main.py --do_train --do_eval
    [--config_train=configs.config_train]
    [--output_dir=output]
```
Here:

- `config_train`: Configurations of GPT-2 training, including data and optimization hyperparameters. By default, the config file [`configs/config_train.py`](configs/config_train.py) is used. Remember to specify correct data path if you are using your own data.
- `output_dir`: The output path where checkpoints are saved.

By default, the GPT-2 `117M` model is used. To use the GPT-2 `345M` model instead, specify relevant FLAGS as below:
```
    python gpt2_train_main.py --do_train --do_eval \
    --config_model=configs.config_model_345M \
    --pretrain_model_dir=gpt2_pretrained_models/model_345M \
    --pretrain_checkpoint=gpt2_pretrained_models/model_345M/model.ckpt \
    [--config_train=configs.config_train]
    [--output_dir=output]
```
where `--pretrain_checkpoint` is necessary only when you want to load the pretrained checkpoint, and is ignored if `--checkpoint` is specified. 

Please see the FLAGS in the code for more options.

For **Multi-GPU training** on one or multiple machines, you may first install the prerequisite OpenMPI and Hovorod packages, as detailed in the [distributed_gpu](https://github.com/asyml/texar/tree/master/examples/distributed_gpu) example. 

Then run the following cmd for training and evaluation. The cmd trains the model on local with 2 GPUs. Evaluation is performed with the single rank-0 GPU.
```
mpirun -np 2 \
    -H  localhost:2\
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl tcp,self \
    -mca btl_tcp_if_include ens3 \
    python gpt2_train_main.py --do_train --do_eval --distributed
    [--config_train=configs.config_train]
    [--output_dir=output]
```
The key configurations of multi-gpu training:

* `-np`: total number of processes
* `-H`: IP addresses of different servers and the number of processes used in each server. For example, `-H 192.168.11.22:1,192.168.33.44:1`
* `-mca`: sets the MPI communication interface. Use the setting specified above to avoid possible multiprocessing and network communication issues.

  - The above configuration uses the `ens3` network interface. If this interface does not work in your environment (e.g., yielding error message `Unknown interfance name`), you may want to use a different interface (Run cmd `ifconfig` to see alternative interfaces in your environment.)

Please refer to [distributed_gpu](https://github.com/asyml/texar/tree/master/examples/distributed_gpu) example for more details of the other multi-gpu configurations.

Make sure to specifiy the `--distributed` flag as above for multi-gpu training.


### Restore and Test

``
python gpt2_train_main.py --do_test --checkpoint=output/model.ckpt
[--config_train=config_train]
[--output_dir=output]
``

The output is by default saved in `output/test_samples.tsv`, where each line contains the context text and the generated continuation (separated with TAB). 


## Other Use Cases

Texar's `TransformerDecoder` (and other RNN-based decoders) easily supports common, advanced, or customized use, such as:

* Sample or continuation generation
* Greedy / (top-k) sample / Gumbel-softmax / beam-search / ... / your-customized decoding
* Training / fine-tuning in (un)conditional settings
* Perplexity evaluation

**For example**, after creating the language model
```python
def _embedding_fn(ids, times):
    return word_embedder(ids) + pos_embedder(times)
    
decoder = TransformerDecoder(
    output_layer=tf.transpose(word_embedder.embedding), 
    hparams=gpt2_hparams)
```
We can do

**Ex. Use 1): Continuation generation w/ greedy decoding**

```python
output, output_length = decoder(
    context=ctx,
    context_sequence_length=ctx_len,
    decoding_strategy='infer_greedy',
    end_token=end_token
    embedding=_embedding_fn)
    
sample_id = output.sample_id
logits = output.logits
```

**Ex. Use 2): Top-k sample decoding**

```python    
topk_helper = tx.modules.TopKSampleEmbeddingHelper(
    embedding=_embedding_fn,
    start_tokens=ctx[:,0],
    end_token=end_token,
    top_k=20,
    softmax_temperature=0.7)
    
output, output_length = decoder(
    context=ctx,
    context_sequence_length=ctx_len,
    helper=topk_helper)
```

**Ex. Use 3): Fine-tuning for conditional generation**

```python
tgt_embed = word_embedder(truth_target[:, :-1]) + pos_embedder(sequence_length=tgt_len-1)

output = decoder(
    memory=source_hidden_states, 
    memory_sequence_length=src_len,
    inputs=tgt_embed,
    decoding_strategy='train_greedy') # teacher-forcing decoding
    
loss = tx.losses.sequence_sparse_softmax_cross_entropy(
    lables=truth_target[:, 1:],
    logits=output.logits,
    sequence_length=tgt_len-1)
```
