#
"""
Various helper classes for RNN decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.seq2seq import TrainingHelper as TFTrainingHelper


class EmbeddingTrainingHelper(TFTrainingHelper):
    """A training helper that uses embeddings.

    Returned sample_ids are the argmax of the RNN output logits.

    Args:
        inputs ((structure of) integer Tensors): Sequences of input token
            indexes.
        sequence_length (1D integer list or Tensor): Lengths of input token
            sequences.
        embedding: A callable that takes a vector tensor of `ids` (argmax ids),
            or the `params` argument for `embedding_lookup` (e.g., the embedding
            Tensor).
        time_major (bool): Whether the tensors in `inputs` are time major.
            If `False` (default), they are assumed to be batch major.
        name (string): Name scope for any created operations.

    Raises:
        ValueError: if `sequence_length` is not a 1D tensor.
    """

    def __init__(self, inputs, # pylint: disable=too-many-arguments
                 sequence_length, embedding, time_major=False, name=None):
        with tf.name_scope(name, "EmbeddingTrainingHelper", [embedding]): # pylint: disable=not-context-manager
            if callable(embedding):
                self._embedding_fn = embedding
            else:
                self._embedding_fn = (
                    lambda ids: tf.nn.embedding_lookup(embedding, ids))
            emb_inputs = self._embedding_fn(inputs)
            TFTrainingHelper.__init__(
                self,
                inputs=emb_inputs,
                sequence_length=sequence_length,
                time_major=time_major,
                name=name)

