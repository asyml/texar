.. role:: hidden
    :class: hidden

Modules
*******

Module Base
===========

.. autoclass:: texar.modules.ModuleBase
    :members:

Networks
========

.. autoclass:: texar.modules.FeedForwardNetwork
    :members:

Encoders
========

.. autoclass:: texar.modules.EncoderBase
    :members:

.. autoclass:: texar.modules.RNNEncoderBase
    :members:

.. autoclass:: texar.modules.UnidirectionalRNNEncoder
    :members:

.. autoclass:: texar.modules.BidirectionalRNNEncoder
    :members:

Decoders
========

RNNDecoderBase
------------------------

.. autoclass:: texar.modules.RNNDecoderBase
    :members:

BasicRNNDecoderOutput
-------------------------------

.. autoclass:: texar.modules.BasicRNNDecoderOutput
    :members:

BasicRNNDecoder
-------------------------

.. autoclass:: texar.modules.BasicRNNDecoder
    :members:

AttentionRNNDecoderOutput
-----------------------------------

.. autoclass:: texar.modules.AttentionRNNDecoderOutput
    :members:

AttentionRNNDecoder
-----------------------------

.. autoclass:: texar.modules.AttentionRNNDecoder
    :members:

default_helper_train_hparams
--------------------------------------

.. autofunction:: texar.modules.default_helper_train_hparams

default_helper_infer_hparams
--------------------------------------

.. autofunction:: texar.modules.default_helper_infer_hparams

get_helper
------------------------

.. autofunction:: texar.modules.get_helper

EmbeddingTrainingHelper
----------------------------------

.. autoclass:: texar.modules.EmbeddingTrainingHelper
    :members:

Connectors
==========

ConnectorBase
--------------------------------------

.. autoclass:: texar.modules.ConnectorBase
    :members:

ConstantConnector
--------------------------------------

.. autoclass:: texar.modules.ConstantConnector
    :members:

ForwardConnector
--------------------------------------

.. autoclass:: texar.modules.ForwardConnector
    :members:

MLPTransformConnector
--------------------------------------

.. autoclass:: texar.modules.MLPTransformConnector
    :members:

ReparameterizedStochasticConnector
--------------------------------------------

.. autoclass:: texar.modules.ReparameterizedStochasticConnector
    :members:

StochasticConnector
--------------------------------------

.. autoclass:: texar.modules.StochasticConnector
    :members:

ConcatConnector
--------------------------------------

.. autoclass:: texar.modules.ConcatConnector
    :members:

Embedders
=========

WordEmbedder
--------------------------------------

.. autoclass:: texar.modules.embedders.WordEmbedder
    :members:

.. autofunction:: texar.modules.embedders.embedder_utils.get_embedding

.. autofunction:: texar.modules.embedders.embedder_utils.default_embedding_hparams

Memory
======

MemNetSingleLayer
--------------------------------------

.. autoclass:: texar.modules.memory.MemNetSingleLayer
    :members:

MemNetBase
--------------------------------------

.. autoclass:: texar.modules.memory.MemNetBase
    :members:

MemNetRNNLike
--------------------------------------

.. autoclass:: texar.modules.memory.MemNetRNNLike
    :members:

default_embedder_fn
--------------------------------------

.. autofunction:: texar.modules.memory.default_embedder_fn

