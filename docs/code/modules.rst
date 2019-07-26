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

:hidden:`MultiheadAttentionEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.MultiheadAttentionEncoder
    :members:

:hidden:`TransformerEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.TransformerEncoder
    :members:

:hidden:`BertEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.BertEncoder
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

:hidden:`XLNetEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.XLNetEncoder
    :members:
    :exclude-members: _execute

:hidden:`default_transformer_poswise_net_hparams`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.modules.default_transformer_poswise_net_hparams

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.AttentionRNNDecoderOutput
    :members:

:hidden:`beam_search_decode`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.modules.beam_search_decode

:hidden:`TransformerDecoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.TransformerDecoder
    :members:

:hidden:`TransformerDecoderOutput`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.TransformerDecoderOutput
    :members:

:hidden:`Helper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.Helper
    :members:

:hidden:`GreedyEmbeddingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.GreedyEmbeddingHelper
    :members:
    :inherited-members:

:hidden:`SampleEmbeddingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.SampleEmbeddingHelper
    :members:
    :inherited-members:

:hidden:`TopKSampleEmbeddingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.TopKSampleEmbeddingHelper
    :members:
    :inherited-members:

:hidden:`SoftmaxEmbeddingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.SoftmaxEmbeddingHelper
    :members:
    :inherited-members:

:hidden:`GumbelSoftmaxEmbeddingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.GumbelSoftmaxEmbeddingHelper
    :members:
    :inherited-members:

:hidden:`TrainingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.TrainingHelper
    :members:
    :inherited-members:

:hidden:`ScheduledEmbeddingTrainingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.ScheduledEmbeddingTrainingHelper
    :members:
    :inherited-members:

:hidden:`ScheduledOutputTrainingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.ScheduledOutputTrainingHelper
    :members:
    :inherited-members:

:hidden:`InferenceHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.InferenceHelper
    :members:
    :inherited-members:

:hidden:`CustomHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.CustomHelper
    :members:
    :inherited-members:

:hidden:`get_helper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.modules.get_helper


Connectors
==========

:hidden:`ConnectorBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.ConnectorBase
    :members:
    :inherited-members:

:hidden:`ConstantConnector`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.ConstantConnector
    :members:
    :inherited-members:

:hidden:`ForwardConnector`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.ForwardConnector
    :members:
    :inherited-members:

:hidden:`MLPTransformConnector`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.MLPTransformConnector
    :members:
    :inherited-members:

:hidden:`ReparameterizedStochasticConnector`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.ReparameterizedStochasticConnector
    :members:
    :inherited-members:

:hidden:`StochasticConnector`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.StochasticConnector
    :members:
    :inherited-members:


Classifiers
============

:hidden:`Conv1DClassifier`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.Conv1DClassifier
    :members:
    :inherited-members:

:hidden:`UnidirectionalRNNClassifier`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.UnidirectionalRNNClassifier
    :members:
    :inherited-members:

:hidden:`BertClassifier`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.BertClassifier
    :members:
    :inherited-members:

:hidden:`XLNetClassifier`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.XLNetClassifier
    :members:

Networks
========

:hidden:`FeedForwardNetworkBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.FeedForwardNetworkBase
    :members:
    :inherited-members:

:hidden:`FeedForwardNetwork`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.FeedForwardNetwork
    :members:
    :inherited-members:

:hidden:`Conv1DNetwork`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.Conv1DNetwork
    :members:
    :inherited-members:

Pre-trained
===========

.. spelling::
    pooler

:hidden:`PretrainedBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.PretrainedBase
    :members:

Regressor
==========

:hidden:`XLNetRegressor`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.XLNetRegressor
    :members:

Memory
======

:hidden:`MemNetBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.MemNetBase
    :members:
    :inherited-members:

:hidden:`MemNetRNNLike`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.MemNetRNNLike
    :members:
    :inherited-members:
    :exclude-members: get_default_embed_fn 

:hidden:`default_memnet_embed_fn_hparams`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.modules.default_memnet_embed_fn_hparams

Policy
=========

:hidden:`PolicyNetBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.PolicyNetBase
    :members:
    :inherited-members:

:hidden:`CategoricalPolicyNet`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.CategoricalPolicyNet
    :members:
    :inherited-members:

Q-Nets
=========

:hidden:`QNetBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.QNetBase
    :members:
    :inherited-members:

:hidden:`CategoricalPolicyNet`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.CategoricalQNet
    :members:
    :inherited-members:

Berts
=========

:hidden:`BertBase`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.modules.BertBase
    :members:
