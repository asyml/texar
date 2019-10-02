.. role:: hidden
    :class: hidden-section

Data
*******

Tokenizers
==========

:hidden:`TokenizerBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.data.TokenizerBase
    :members:

:hidden:`BERTTokenizer`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.data.BERTTokenizer
    :members:

:hidden:`XLNetTokenizer`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.data.XLNetTokenizer
    :members:

Vocabulary
==========

:hidden:`SpecialTokens`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.tf.data.SpecialTokens
    :members:

:hidden:`Vocab`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.tf.data.Vocab
    :members:

Embedding
==========

:hidden:`Embedding`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.tf.data.Embedding
    :members:

:hidden:`load_word2vec`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.data.load_word2vec

:hidden:`load_glove`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: texar.tf.data.load_glove

Data
==========

:hidden:`DataBase`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.tf.data.DataBase
    :members:

:hidden:`MonoTextData`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.tf.data.MonoTextData
    :members:
    :inherited-members:
    :exclude-members: make_vocab,make_embedding

:hidden:`PairedTextData`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.tf.data.PairedTextData
    :members:
    :inherited-members:
    :exclude-members: make_vocab,make_embedding

:hidden:`ScalarData`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.tf.data.ScalarData
    :members:
    :inherited-members:

:hidden:`TFRecordData`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.tf.data.TFRecordData
    :members:
    :inherited-members:


:hidden:`MultiAlignedData`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.tf.data.MultiAlignedData
    :members:
    :inherited-members:
    :exclude-members: make_vocab,make_embedding

:hidden:`TextDataBase`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.tf.data.TextDataBase
    :members:


Data Iterators
===============

:hidden:`DataIteratorBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.tf.data.DataIteratorBase
    :members:

:hidden:`DataIterator`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.tf.data.DataIterator
    :members:

:hidden:`TrainTestDataIterator`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.tf.data.TrainTestDataIterator
    :members:


:hidden:`FeedableDataIterator`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.tf.data.FeedableDataIterator
    :members:


:hidden:`TrainTestFeedableDataIterator`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.tf.data.TrainTestFeedableDataIterator
    :members:

Data Utils
==========

:hidden:`random_shard_dataset`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.data.random_shard_dataset

:hidden:`maybe_tuple`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.data.maybe_tuple

:hidden:`make_partial`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.data.make_partial

:hidden:`maybe_download`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.data.maybe_download

:hidden:`read_words`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.data.read_words

:hidden:`make_vocab`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.data.make_vocab

:hidden:`count_file_lines`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.data.count_file_lines

:hidden:`make_chained_transformation`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.data.make_chained_transformation

:hidden:`make_combined_transformation`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.data.make_combined_transformation
