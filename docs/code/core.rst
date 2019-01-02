.. role:: hidden
    :class: hidden-section

Core
****


Cells
=====

:hidden:`default_rnn_cell_hparams`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.default_rnn_cell_hparams 

:hidden:`get_rnn_cell`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_rnn_cell

:hidden:`get_rnn_cell_trainable_variables`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_rnn_cell_trainable_variables

Layers
======

:hidden:`get_layer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_layer

:hidden:`MaxReducePooling1D`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.MaxReducePooling1D
    :members:

:hidden:`AverageReducePooling1D`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.AverageReducePooling1D
    :members:

:hidden:`get_pooling_layer_hparams`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_pooling_layer_hparams

:hidden:`MergeLayer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.MergeLayer
    :members:

:hidden:`SequentialLayer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.SequentialLayer
    :members:

:hidden:`default_regularizer_hparams`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.default_regularizer_hparams

:hidden:`get_regularizer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_regularizer

:hidden:`get_initializer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_initializer

:hidden:`get_activation_fn`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_activation_fn

:hidden:`get_constraint_fn`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_constraint_fn

:hidden:`default_conv1d_kwargs`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.default_conv1d_kwargs

:hidden:`default_dense_kwargs`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.default_dense_kwargs


Optimization
=============

:hidden:`default_optimization_hparams`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.default_optimization_hparams

:hidden:`get_train_op`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_train_op

:hidden:`get_optimizer_fn`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_optimizer_fn

:hidden:`get_optimizer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_optimizer

:hidden:`get_learning_rate_decay_fn`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_learning_rate_decay_fn

:hidden:`get_gradient_clip_fn`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.core.get_gradient_clip_fn


Exploration
============

:hidden:`EpsilonLinearDecayExploration`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.EpsilonLinearDecayExploration
    :members:

:hidden:`ExplorationBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.ExplorationBase
    :members:

Replay Memories
================

:hidden:`DequeReplayMemory`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.DequeReplayMemory
    :members:

:hidden:`ReplayMemoryBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.core.ReplayMemoryBase
    :members:
