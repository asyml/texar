.. role:: hidden
    :class: hidden-section

Core
****


Cells
=====

:hidden:`default_rnn_cell_hparams`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.core.default_rnn_cell_hparams

:hidden:`get_rnn_cell`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.core.get_rnn_cell

:hidden:`get_rnn_cell_trainable_variables`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.core.get_rnn_cell_trainable_variables

Layers
======

:hidden:`get_layer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.core.get_layer

:hidden:`MaxReducePooling1D`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.core.MaxReducePooling1D
    :members:

:hidden:`AverageReducePooling1D`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.core.AverageReducePooling1D
    :members:

:hidden:`get_pooling_layer_hparams`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.core.get_pooling_layer_hparams

:hidden:`MergeLayer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.core.MergeLayer
    :members:

:hidden:`SequentialLayer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.core.SequentialLayer
    :members:

:hidden:`default_regularizer_hparams`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.core.default_regularizer_hparams

:hidden:`get_regularizer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.core.get_regularizer

:hidden:`get_initializer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.core.get_initializer

:hidden:`get_activation_fn`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.core.get_activation_fn

:hidden:`get_constraint_fn`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.core.get_constraint_fn

:hidden:`default_conv1d_kwargs`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.core.default_conv1d_kwargs

:hidden:`default_dense_kwargs`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.core.default_dense_kwargs


Optimization
=============

:hidden:`default_optimization_hparams`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.core.default_optimization_hparams

:hidden:`get_train_op`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.core.get_train_op

:hidden:`get_optimizer_fn`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.core.get_optimizer_fn

:hidden:`get_optimizer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.core.get_optimizer

:hidden:`get_learning_rate_decay_fn`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.core.get_learning_rate_decay_fn

:hidden:`get_gradient_clip_fn`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.core.get_gradient_clip_fn


Exploration
============

:hidden:`EpsilonLinearDecayExploration`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.core.EpsilonLinearDecayExploration
    :members:

:hidden:`ExplorationBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.core.ExplorationBase
    :members:

Replay Memories
================

:hidden:`DequeReplayMemory`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.core.DequeReplayMemory
    :members:

:hidden:`ReplayMemoryBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.core.ReplayMemoryBase
    :members:
