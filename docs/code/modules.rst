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

:hidden:`TransformerEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.TransformerEncoder
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

:hidden:`RNNDecoderBase`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.RNNDecoderBase
    :members:
    :inherited-members:
    :exclude-members: initialize,step,finalize,tracks_own_finished,output_size,output_dtype

:hidden:`BasicRNNDecoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.BasicRNNDecoder
    :members:
    :inherited-members:
    :exclude-members: initialize,step,finalize,tracks_own_finished,output_size,output_dtype

:hidden:`BasicRNNDecoderOutput`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.BasicRNNDecoderOutput
    :members:

:hidden:`AttentionRNNDecoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.AttentionRNNDecoder
    :members:
    :inherited-members:
    :exclude-members: initialize,step,finalize,tracks_own_finished,output_size,output_dtype

:hidden:`AttentionRNNDecoderOutput`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.AttentionRNNDecoderOutput
    :members:

:hidden:`beam_search_decode`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.modules.beam_search_decode

:hidden:`SoftmaxEmbeddingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.SoftmaxEmbeddingHelper
    :members:

:hidden:`GumbelSoftmaxEmbeddingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.GumbelSoftmaxEmbeddingHelper
    :members:

:hidden:`get_helper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.modules.get_helper


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

