
## [Unreleased]

### New features

* `GPT-2`: OpenAI pretrained language model. ([#91](https://github.com/asyml/texar/pull/91), [example](https://github.com/asyml/texar/tree/master/examples/gpt-2))
* `TopKSampleEmbeddingHelper` to perform top_k random sample decoding. ([baa09ff](https://github.com/asyml/texar/commit/baa09ff0ec898996d7be2535e73bedd1e92c1fb2))

### Feature improvements

* `TransformerDecoder` supports `helper` arguments to specify decoding strategy. ([#76](https://github.com/asyml/texar/pull/76))

### Fixes

* Fix error when `beam_search_decode` with `output_layer=tf.identity` ([#77](https://github.com/asyml/texar/pull/77))
* Fix readthedocs compilation error ([#85](https://github.com/asyml/texar/pull/85))

## [v0.1.0](https://github.com/asyml/texar/releases/tag/v0.1.0) (2019-02-06)

### New features

* [2019-01-02] Support distributed-GPU training. See the [example](https://github.com/asyml/texar/tree/master/examples/distributed_gpu) 
* [2018-11-29] Support pre-trained BERT model. See the [example](https://github.com/asyml/texar/tree/master/examples/bert) 
