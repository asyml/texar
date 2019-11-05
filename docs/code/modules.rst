.. role:: hidden
    :class: hidden-section

Modules
*******

ModuleBase
===========

.. autoclass:: texar.tf.ModuleBase
    :members:

Embedders
=========

:hidden:`WordEmbedder`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.WordEmbedder
    :members:

:hidden:`PositionEmbedder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.PositionEmbedder
    :members:

:hidden:`SinusoidsPositionEmbedder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.SinusoidsPositionEmbedder
    :members:

:hidden:`EmbedderBase`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.EmbedderBase
    :members:


Encoders
========

:hidden:`UnidirectionalRNNEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.UnidirectionalRNNEncoder
    :members:

:hidden:`BidirectionalRNNEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.BidirectionalRNNEncoder
    :members:

:hidden:`HierarchicalRNNEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.HierarchicalRNNEncoder
    :members:

:hidden:`MultiheadAttentionEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.MultiheadAttentionEncoder
    :members:

:hidden:`TransformerEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.TransformerEncoder
    :members:

:hidden:`BERTEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.BERTEncoder
    :members:

:hidden:`Conv1DEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.Conv1DEncoder
    :members:

:hidden:`EncoderBase`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.EncoderBase
    :members:

:hidden:`RNNEncoderBase`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.RNNEncoderBase
    :members:

:hidden:`XLNetEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.XLNetEncoder
    :members:
    :exclude-members: _execute

:hidden:`default_transformer_poswise_net_hparams`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.modules.default_transformer_poswise_net_hparams


Decoders
========

:hidden:`RNNDecoderBase`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.RNNDecoderBase
    :members:
    :inherited-members:
    :exclude-members: initialize,step,finalize,tracks_own_finished,output_size,output_dtype

:hidden:`BasicRNNDecoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.BasicRNNDecoder
    :members:
    :inherited-members:
    :exclude-members: initialize,step,finalize,tracks_own_finished,output_size,output_dtype

:hidden:`BasicRNNDecoderOutput`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.BasicRNNDecoderOutput
    :members:

:hidden:`AttentionRNNDecoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.AttentionRNNDecoder
    :members:
    :inherited-members:
    :exclude-members: initialize,step,finalize,tracks_own_finished,output_size,output_dtype

:hidden:`AttentionRNNDecoderOutput`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.AttentionRNNDecoderOutput
    :members:

:hidden:`GPT2Decoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.GPT2Decoder
    :members:

:hidden:`beam_search_decode`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.modules.beam_search_decode

:hidden:`TransformerDecoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.TransformerDecoder
    :members:

:hidden:`TransformerDecoderOutput`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.TransformerDecoderOutput
    :members:

:hidden:`Helper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.Helper
    :members:

:hidden:`GreedyEmbeddingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.GreedyEmbeddingHelper
    :members:
    :inherited-members:

:hidden:`SampleEmbeddingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.SampleEmbeddingHelper
    :members:
    :inherited-members:

:hidden:`TopKSampleEmbeddingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.TopKSampleEmbeddingHelper
    :members:
    :inherited-members:

:hidden:`SoftmaxEmbeddingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.SoftmaxEmbeddingHelper
    :members:
    :inherited-members:

:hidden:`GumbelSoftmaxEmbeddingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.GumbelSoftmaxEmbeddingHelper
    :members:
    :inherited-members:

:hidden:`TrainingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.TrainingHelper
    :members:
    :inherited-members:

:hidden:`ScheduledEmbeddingTrainingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.ScheduledEmbeddingTrainingHelper
    :members:
    :inherited-members:

:hidden:`ScheduledOutputTrainingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.ScheduledOutputTrainingHelper
    :members:
    :inherited-members:

:hidden:`InferenceHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.InferenceHelper
    :members:
    :inherited-members:

:hidden:`CustomHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.CustomHelper
    :members:
    :inherited-members:

:hidden:`get_helper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.modules.get_helper


Classifiers
============

:hidden:`Conv1DClassifier`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.Conv1DClassifier
    :members:
    :inherited-members:

:hidden:`UnidirectionalRNNClassifier`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.UnidirectionalRNNClassifier
    :members:
    :inherited-members:

:hidden:`BertClassifier`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.BERTClassifier
    :members:
    :inherited-members:

:hidden:`XLNetClassifier`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.XLNetClassifier
    :members:


Regressors
==========

:hidden:`XLNetRegressor`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.XLNetRegressor
    :members:


Pre-trained
===========

:hidden:`PretrainedMixin`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.PretrainedMixin
    :members:
    :private-members:
    :exclude-members: _name_to_variable

:hidden:`PretrainedBERTMixin`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.PretrainedBERTMixin
    :members:

:hidden:`PretrainedXLNetMixin`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.PretrainedXLNetMixin
    :members:


Connectors
==========

:hidden:`ConnectorBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.ConnectorBase
    :members:
    :inherited-members:

:hidden:`ConstantConnector`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.ConstantConnector
    :members:
    :inherited-members:

:hidden:`ForwardConnector`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.ForwardConnector
    :members:
    :inherited-members:

:hidden:`MLPTransformConnector`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.MLPTransformConnector
    :members:
    :inherited-members:

:hidden:`ReparameterizedStochasticConnector`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.ReparameterizedStochasticConnector
    :members:
    :inherited-members:

:hidden:`StochasticConnector`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.StochasticConnector
    :members:
    :inherited-members:


Networks
========

:hidden:`FeedForwardNetworkBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.FeedForwardNetworkBase
    :members:
    :inherited-members:

:hidden:`FeedForwardNetwork`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.FeedForwardNetwork
    :members:
    :inherited-members:

:hidden:`Conv1DNetwork`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.Conv1DNetwork
    :members:
    :inherited-members:


Memory
======

:hidden:`MemNetBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.MemNetBase
    :members:
    :inherited-members:

:hidden:`MemNetRNNLike`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.MemNetRNNLike
    :members:
    :inherited-members:
    :exclude-members: get_default_embed_fn 

:hidden:`default_memnet_embed_fn_hparams`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.modules.default_memnet_embed_fn_hparams

Policy
=========

:hidden:`PolicyNetBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.PolicyNetBase
    :members:
    :inherited-members:

:hidden:`CategoricalPolicyNet`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.CategoricalPolicyNet
    :members:
    :inherited-members:

Q-Nets
=========

:hidden:`QNetBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.QNetBase
    :members:
    :inherited-members:

:hidden:`CategoricalPolicyNet`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.modules.CategoricalQNet
    :members:
    :inherited-members:
