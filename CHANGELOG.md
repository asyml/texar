
## [Unreleased]

### New features

### Feature improvements

### Fixes

* Fix docstring of connector `_mlp_transform`. ([#192](https://github.com/asyml/texar/pull/192))

## [v0.2.2](https://github.com/asyml/texar/releases/tag/v0.2.2) (2019-08-05)

### New features

* Enable installation from [Pypi](https://pypi.org/project/texar/). ([#186](https://github.com/asyml/texar/pull/186))

### Feature improvements

* Use lazy import to be compatible with [texar-pytorch](https://github.com/asyml/texar-pytorch). ([#183](https://github.com/asyml/texar/pull/183)) 

### Fixes

## [v0.2.1](https://github.com/asyml/texar/releases/tag/v0.2.1) (2019-07-28)

### New features

* Add support for GPT-2 345M model in [examples/gpt-2](https://github.com/asyml/texar/tree/master/examples/gpt-2). ([#156](https://github.com/asyml/texar/pull/156)) 
* Add BERT modules, including `texar.modules.BERTEncoder` ([doc](https://texar.readthedocs.io/en/latest/code/modules.html#texar.modules.BertEncoder)) and `texar.modules.BERTClassifier` ([doc](https://texar.readthedocs.io/en/latest/code/modules.html#bertclassifierv)). ([#167](https://github.com/asyml/texar/pull/167))

### Feature improvements

* Refactor `TransformerEncoder` and `TransformerDecoder` to separate position embeddings from the modules. ([#126](https://github.com/asyml/texar/pull/126))
* Allow passing a Tensor to `output_layer` of decoders' constructors -- used for weight tie b/w the output layer and input embedding matrix.  ([#126](https://github.com/asyml/texar/pull/126))
* `TransformerDecoder` constructor interface made exact the same with `RNN decoders` constructor interfaces. ([#126](https://github.com/asyml/texar/pull/126))
* Refactor decoder `Helper`s to allow two-argument `embedding_fn` (supporting for position embedding). ([#126](https://github.com/asyml/texar/pull/126))
* Refactor `SinusoidsPositionEmbedder` to enable infinite large or negative position indexes. ([#176](https://github.com/asyml/texar/pull/176))

### Fixes

* Fix `texar.losses.reduce_batch_time` when `sequence` has dtype other than `tf.float32`. ([#143](https://github.com/asyml/texar/issues/143))
* Fix `texar.losses.reduce_dimensions` when `average_axes` or `sum_axes` is `int`. ([#141](https://github.com/asyml/texar/pull/141))
* Fix [GPT-2](https://github.com/asyml/texar/tree/master/examples/gpt-2) tokenization loading path. ([#165](https://github.com/asyml/texar/pull/165))
* Fix [examples/vae_text](https://github.com/asyml/texar/tree/master/examples/vae_text) EOS bug. ([#168](https://github.com/asyml/texar/pull/168)) 
* Fix transformer [bleu_tool.py](https://github.com/asyml/texar/blob/master/examples/transformer/bleu_tool.py) when `translation_length` is 0. ([#176](https://github.com/asyml/texar/pull/176))
* Fix `StochasticConnector` and `ReparameterizedStochasticConnector` when `transform=False`. ([#179](https://github.com/asyml/texar/pull/179))

## [v0.2.0](https://github.com/asyml/texar/releases/tag/v0.2.0) (2019-04-09)

### New features

* `TFRecordData`: A new data module for reading and processing TFRecord data, with support for, e.g., image data, feature data, etc. ([#107](https://github.com/asyml/texar/pull/107))
* `GPT-2`: OpenAI pretrained language model. ([#91](https://github.com/asyml/texar/pull/91), [example](https://github.com/asyml/texar/tree/master/examples/gpt-2))
* `TopKSampleEmbeddingHelper` to perform top_k random sample decoding. ([baa09ff](https://github.com/asyml/texar/commit/baa09ff0ec898996d7be2535e73bedd1e92c1fb2))

### Feature improvements

* Refactor [`BERT` example](https://github.com/asyml/texar/tree/master/examples/bert) using `TFRecordData` data module. 
* `TransformerDecoder` supports `helper` arguments to specify decoding strategy. ([#76](https://github.com/asyml/texar/pull/76))

### Fixes

* Fix variable collection bug in [`examples/seqgan`](https://github.com/asyml/texar/tree/master/examples/seqgan). ([#110](https://github.com/asyml/texar/pull/110))
* Fix error when `beam_search_decode` with `output_layer=tf.identity` ([#77](https://github.com/asyml/texar/pull/77))
* Fix readthedocs compilation error ([#85](https://github.com/asyml/texar/pull/85))

## [v0.1.0](https://github.com/asyml/texar/releases/tag/v0.1.0) (2019-02-06)

### New features

* [2019-01-02] Support distributed-GPU training. See the [example](https://github.com/asyml/texar/tree/master/examples/distributed_gpu) 
* [2018-11-29] Support pre-trained BERT model. See the [example](https://github.com/asyml/texar/tree/master/examples/bert) 
