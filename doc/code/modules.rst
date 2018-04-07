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

:hidden:`RNNDecoderBase`
------------------------

.. autoclass:: texar.modules.RNNDecoderBase
    :members:

:hidden:`BasicRNNDecoderOutput`
-------------------------------

.. autoclass:: texar.modules.BasicRNNDecoderOutput
    :members:

:hidden:`BasicRNNDecoder`
-------------------------

.. autoclass:: texar.modules.BasicRNNDecoder
    :members:

:hidden:`AttentionRNNDecoderOutput`
-----------------------------------

.. autoclass:: texar.modules.AttentionRNNDecoderOutput
    :members:

:hidden:`AttentionRNNDecoder`
-----------------------------

.. autoclass:: texar.modules.AttentionRNNDecoder
    :members:

:hidden:`default_helper_train_hparams`
--------------------------------------

.. autofunction:: texar.modules.default_helper_train_hparams

:hidden:`default_helper_infer_hparams`
--------------------------------------

.. autofunction:: texar.modules.default_helper_infer_hparams

:hidden:`get_helper`
------------------------

.. autofunction:: texar.modules.get_helper

:hidden:`EmbeddingTrainingHelper`
----------------------------------

.. autoclass:: texar.modules.EmbeddingTrainingHelper
    :members:

Connectors
==========

:hidden:`ConnectorBase`
--------------------------------------

.. autoclass:: texar.modules.ConnectorBase
    :members:

:hidden:`ConstantConnector`
--------------------------------------

.. autoclass:: texar.modules.ConstantConnector
    :members:

:hidden:`ForwardConnector`
--------------------------------------

.. autoclass:: texar.modules.ForwardConnector
    :members:

:hidden:`MLPTransformConnector`
--------------------------------------

.. autoclass:: texar.modules.MLPTransformConnector
    :members:

:hidden:`ReparameterizedStochasticConnector`
--------------------------------------------

.. autoclass:: texar.modules.ReparameterizedStochasticConnector
    :members:

:hidden:`StochasticConnector`
--------------------------------------

.. autoclass:: texar.modules.StochasticConnector
    :members:

:hidden:`ConcatConnector`
--------------------------------------

.. autoclass:: texar.modules.ConcatConnector
    :members:
