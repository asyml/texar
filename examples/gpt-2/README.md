# GPT-2: Pre-trained Langauge Model

This is a Texar implementation of [OpenAI GPT-2 (Generative Pre-Trainning)](https://github.com/openai/gpt-2) language model, which allows to load official pre-trained model parameters, generate samples, etc.

With Texar, building the GPT-2 model is as simple as creating a [`TransformerDecoder`](https://texar.readthedocs.io/en/latest/code/modules.html#transformerdecoder) instance. We can initialize the parameters of the TransformerDecoder using a pre-trained GPT-2 checkpoint by calling `init_gpt2_checkpoint(path_to_gpt2_checkpoint)` .

In sum, this example showcases:

* Contructing and using pre-trained GPT-2 models in Texar
* Using GPT-2 to generate text samples with or without context
* Examples of other use cases

## Quick Start
### Download GPT-2 Pre-trained Model

Download the GPT-2 model checkpoint with the following command:
```
sh gpt2_pretrained_models/download_model.sh 117M
```
By default, it will download a pretrained model named `117M` to `gpt2_pretrained_models/`.

### Usage
| WARNING: Samples are unfiltered and may contain offensive content. |
| --- |

#### Interactive mode (to generate samples with context)

This mode will initialize an interactive interface, which allows users to type in the context sentence. The model then generates continuation of the context. Top-K sample decoding is used.

```
python generative_pretraining_main.py --is_interactive \
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
python generative_pretraining_main.py
--nsamples=1 \
--batch_size=1 \
--max_decoding_len=100 \
--temperature=0.7 \
--top_k=40
```

Here:

- `nsamples`: Total number of samples to generate, must be dividable by the `batch_size`.
- `batch_size`: Each iteration generates `batch_size` number of samples.

**Example output:**

```
"A new government and a healthy economy have a chance to take this up."

After he said the election's outcome in the House was important and had helped to build 
confidence in the House, former Ukip leader Nigel Farage spoke about working to boost 
the economy, saying the vote for the "lefties" and others "were bad optics for Labour 
in this way".
```

## Other Use Cases

Texar's `TransformerDecoder` (and other RNN-based decoders) easily supports common, advanced, or customized use, such as:

* Sample or continuation generation
* Greedy / (top-k) sample / Gumbel-softmax / beam-search / ... / your-customized decoding
* Training / fine-tuning in (un)conditional settings
* Perplexity evaluation

**For example**, after creating the language model
```python
decoder = TransformerDecoder(embedder, hparams=gpt2_hparams)
```
We can do

**Ex. Use 1): Continuation generation w/ greedy decoding**

```python
output, output_length = decoder(
    context=ctx,
    context_sequence_length=ctx_len,
    decoding_strategy='infer_greedy',
    end_token=end_token)
    
sample_id = output.sample_id
logits = output.logits
```

**Ex. Use 2): Top-k sample decoding**

```python
topk_helper = tx.modules.TopKSampleEmbeddingHelper(
    embedding=embedder,
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
output = decoder(
    memory=source_hidden_states, 
    memory_sequence_length=src_len,
    inputs=truth_target[:, :-1],
    sequence_length=tgt_len-1,
    decoding_strategy='train_greedy')
    
loss = tx.losses.sequence_sparse_softmax_cross_entropy(
    lables=truth_target[:, 1:],
    logits=output.logits,
    sequence_length=tgt_len-1)
```
