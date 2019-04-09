
## [Unreleased]

### New features

### Feature improvements

* Refactor `TransformerEncoder` and `TransformerDecoder` to separate position embeddings from the modules. ([#126](https://github.com/asyml/texar/pull/126))
* Allow passing a Tensor to `output_layer` of decoders' constructors -- used for weight tie b/w the output layer and input embedding matrix.  ([#126](https://github.com/asyml/texar/pull/126))
* `TransformerDecoder` constructor interface made exact the same with `RNN decoders` constructor interfaces. ([#126](https://github.com/asyml/texar/pull/126))
* Refactor decoder `Helper`s to allow two-argument `embedding_fn` (supporting for position embedding). ([#126](https://github.com/asyml/texar/pull/126))

### Fixes

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
