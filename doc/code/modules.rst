.. role:: hidden
    :class: hidden

Modules
*******

Module Base
===========

.. autoclass:: txtgen.modules.ModuleBase
    :members:

Encoders
========

.. autoclass:: txtgen.modules.EncoderBase
    :members:

.. autoclass:: txtgen.modules.RNNEncoderBase
    :members:

.. autoclass:: txtgen.modules.ForwardRNNEncoder
    :members:

Decoders
========

:hidden:`RNNDecoderBase`
------------------------

.. autoclass:: txtgen.modules.RNNDecoderBase
    :members:

:hidden:`BasicRNNDecoderOutput`
-------------------------------

.. autoclass:: txtgen.modules.BasicRNNDecoderOutput
    :members:

:hidden:`BasicRNNDecoder`
-------------------------

.. autoclass:: txtgen.modules.BasicRNNDecoder
    :members:

:hidden:`AttentionRNNDecoderOutput`
-----------------------------------

.. autoclass:: txtgen.modules.AttentionRNNDecoderOutput
    :members:

:hidden:`AttentionRNNDecoder`
-----------------------------

.. autoclass:: txtgen.modules.AttentionRNNDecoder
    :members:

:hidden:`default_helper_train_hparams`
--------------------------------------

.. autofunction:: txtgen.modules.default_helper_train_hparams

:hidden:`default_helper_infer_hparams`
--------------------------------------

.. autofunction:: txtgen.modules.default_helper_infer_hparams

:hidden:`get_helper`
------------------------

.. autofunction:: txtgen.modules.get_helper

:hidden:`EmbeddingTrainingHelper`
----------------------------------

.. autoclass:: txtgen.modules.EmbeddingTrainingHelper
    :members:

Connectors
==========

:hidden:`ConnectorBase`
--------------------------------------

.. autoclass:: txtgen.modules.ConnectorBase
    :members:

:hidden:`ConstantConnector`
--------------------------------------

.. autoclass:: txtgen.modules.ConstantConnector
    :members:

:hidden:`ForwardConnector`
--------------------------------------

.. autoclass:: txtgen.modules.ForwardConnector
    :members:

:hidden:`MLPTransformConnector`
--------------------------------------

.. autoclass:: txtgen.modules.MLPTransformConnector
    :members:

:hidden:`ReparameterizedStochasticConnector`
--------------------------------------------

.. autoclass:: txtgen.modules.ReparameterizedStochasticConnector
    :members:

:hidden:`StochasticConnector`
--------------------------------------

.. autoclass:: txtgen.modules.StochasticConnector
    :members:

:hidden:`ConcatConnector`
--------------------------------------

.. autoclass:: txtgen.modules.ConcatConnector
    :members:
