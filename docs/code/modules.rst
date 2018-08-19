.. role:: hidden
    :class: hidden-section

Modules
*******

ModuleBase
===========

.. autoclass:: texar.ModuleBase
    :members:

Embedders
=========

:hidden:`WordEmbedder`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.WordEmbedder
    :members:

:hidden:`PositionEmbedder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.PositionEmbedder
    :members:

:hidden:`SinusoidsPositionEmbedder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.SinusoidsPositionEmbedder
    :members:

:hidden:`EmbedderBase`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.EmbedderBase
    :members:


Encoders
========

:hidden:`UnidirectionalRNNEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.UnidirectionalRNNEncoder
    :members:

:hidden:`BidirectionalRNNEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.BidirectionalRNNEncoder
    :members:

:hidden:`HierarchicalRNNEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.HierarchicalRNNEncoder
    :members:

:hidden:`Conv1DEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.Conv1DEncoder
    :members:

:hidden:`EncoderBase`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.EncoderBase
    :members:

:hidden:`RNNEncoderBase`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.RNNEncoderBase
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

Networks
========

.. autoclass:: texar.modules.FeedForwardNetwork
    :members:

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

