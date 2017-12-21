#
"""
Various helper classes and utilities for RNN decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.seq2seq import TrainingHelper as TFTrainingHelper

from texar.core import utils

# pylint: disable=not-context-manager, too-many-arguments

__all__ = [
    "default_helper_train_hparams", "default_helper_infer_hparams",
    "get_helper", "EmbeddingTrainingHelper"
]

def default_helper_train_hparams():
    """Returns default hyperparameters of an RNN decoder helper in the training
    phase.

    See also :meth:`~texar.modules.decoders.rnn_decoder_helpers.get_helper`
    for information of the hyperparameters.

    Returns:
        dict: A dictionary with following structure and values:

        .. code-block:: python

            {
                # The `helper_type` argument for `get_helper`, i.e., the name
                # or full path to the helper class.
                "type": "EmbeddingTrainingHelper",

                # The `**kwargs` argument for `get_helper`, i.e., additional
                # keyword arguments for constructing the helper.
                "kwargs": {}
            }
    """
    return {
        "type": "EmbeddingTrainingHelper",
        "kwargs": {}
    }

def default_helper_infer_hparams():
    """Returns default hyperparameters of an RNN decoder helper in the inference
    phase.

    See also :meth:`~texar.modules.decoders.rnn_decoder_helpers.get_helper`
    for information of the hyperparameters.

    Returns:
        dict: A dictionary with following structure and values:

        .. code-block:: python

            {
                # The `helper_type` argument for `get_helper`, i.e., the name
                # or full path to the helper class.
                "type": "SampleEmbeddingHelper",

                # The `**kwargs` argument for `get_helper`, i.e., additional
                # keyword arguments for constructing the helper.
                "kwargs": {}
            }
    """
    return {
        "type": "SampleEmbeddingHelper",
        "kwargs": {}
    }


def get_helper(helper_type,
               inputs=None,
               sequence_length=None,
               embedding=None,
               start_tokens=None,
               end_token=None,
               **kwargs):
    """Creates a Helper instance.

    Args:
        helper_type (str): The name or full path to the helper class.
            E.g., the classname of the built-in helpers in
            :mod:`texar.modules.decoders.rnn_decoder_helpers` or
            :mod:`tensorflow.contrib.seq2seq`, or the classname of user-defined
            helpers in :mod:`texar.custom`, or a full path like
            "my_module.MyHelper".
        inputs ((structure of) Tensors, optional): Inputs to the decoder.
        sequence_length (1D integer array or Tensor, optional): Lengths of input
            token sequences.
        embedding (optional): A callable that takes a vector tensor of integer
            indexes, or the `params` argument for `embedding_lookup` (e.g.,
            the embedding Tensor).
        start_tokens (int array or 1D int Tensor, optional): Of shape
            `[batch_size]`. The start tokens.
        end_token (int or int scalar Tensor, optional): The token that marks
            end of decoding.
        **kwargs: additional keyword arguments for constructing the helper.

    Returns:
        An instance of specified helper.
    """
    module_paths = [
        'texar.modules.decoders.rnn_decoder_helpers',
        'tensorflow.contrib.seq2seq',
        'texar.custom']
    class_kwargs = {"inputs": inputs,
                    "sequence_length": sequence_length,
                    "embedding": embedding,
                    "start_tokens": start_tokens,
                    "end_token": end_token}
    class_kwargs.update(kwargs)
    return utils.get_instance_with_redundant_kwargs(
        helper_type, class_kwargs, module_paths)


class EmbeddingTrainingHelper(TFTrainingHelper):
    """A training helper that uses embeddings.

    Returned sample_ids are the argmax of the RNN output logits.

    Args:
        inputs ((structure of) integer Tensors): Sequences of input token
            indexes.
        sequence_length (1D integer list or Tensor): Lengths of input token
            sequences.
        embedding: A callable that takes a vector tensor of integer indexes,,
            or the `params` argument for `embedding_lookup` (e.g., the embedding
            Tensor).
        time_major (bool): Whether the tensors in `inputs` are time major.
            If `False` (default), they are assumed to be batch major.
        name (string): Name scope for any created operations.

    Raises:
        ValueError: if `sequence_length` is not a 1D tensor.
    """

    def __init__(self, inputs, sequence_length, embedding,
                 time_major=False, name=None):
        with tf.name_scope(name, "EmbeddingTrainingHelper", [embedding]):
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
