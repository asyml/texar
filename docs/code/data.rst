.. role:: hidden
    :class: hidden-section

Data
*******

Vocabulary
==========

:hidden:`SpecialTokens`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.data.SpecialTokens
    :members:

:hidden:`Vocab`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.data.Vocab
    :members:

Embedding
==========

:hidden:`Embedding`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.data.Embedding
    :members:

:hidden:`load_word2vec`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.data.load_word2vec

:hidden:`load_glove`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: texar.data.load_glove

Data
==========

:hidden:`DataBase`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.data.DataBase
    :members:

:hidden:`MonoTextData`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.data.MonoTextData
    :members:
    :inherited-members:
    :exclude-members: make_vocab,make_embedding

:hidden:`PairedTextData`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.data.PairedTextData
    :members:
    :inherited-members:
    :exclude-members: make_vocab,make_embedding

:hidden:`ScalarData`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.data.ScalarData
    :members:
    :inherited-members:

:hidden:`MultiAlignedData`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.data.MultiAlignedData
    :members:
    :inherited-members:
    :exclude-members: make_vocab,make_embedding

:hidden:`TextDataBase`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.data.TextDataBase
    :members:


Data Iterators
===============

:hidden:`DataIteratorBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.data.DataIteratorBase
    :members:

:hidden:`DataIterator`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.data.DataIterator
    :members:

:hidden:`TrainTestDataIterator`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.data.TrainTestDataIterator
    :members:


:hidden:`FeedableDataIterator`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.data.FeedableDataIterator
    :members:


:hidden:`TrainTestFeedableDataIterator`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.data.TrainTestFeedableDataIterator
    :members:

Data Utils
==========

:hidden:`random_shard_dataset`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.data.random_shard_dataset

:hidden:`maybe_tuple`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.data.maybe_tuple

:hidden:`make_partial`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.data.make_partial

:hidden:`maybe_download`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.data.maybe_download

:hidden:`read_words`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.data.read_words

:hidden:`make_vocab`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.data.make_vocab

:hidden:`count_file_lines`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.data.count_file_lines

:hidden:`make_chained_transformation`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.data.make_chained_transformation

:hidden:`make_combined_transformation`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.data.make_combined_transformation
