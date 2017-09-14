#
"""
Various helper classes and utilities for RNN decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.seq2seq import TrainingHelper as TFTrainingHelper

from txtgen.core import utils


def make_helper(helper_type,    # pylint: disable=too-many-arguments
                inputs=None,
                sequence_length=None,
                embedding=None,
                start_token=None,
                end_token=None,
                **kwargs):
    """Creates a Helper instance.

    Args:
        helper_type (str): The name or full path to the helper class.
            E.g., the classname of the built-in helpers in
            :mod:`txtgen.modules.decoders.rnn_decoder_helpers` or
            :mod:`tensorflow.contrib.seq2seq`, or the classname of user-defined
            helpers in :mod:`txtgen.custom`, or a full path like
            "my_module.MyHelper".
        inputs ((structure of) Tensors, optional): Inputs to the decoder.
        sequence_length (1D integer list or Tensor, optional): Lengths of input
            token sequences.
        embedding (optional): A callable that takes a vector tensor of integer
            indexes, or the `params` argument for `embedding_lookup` (e.g.,
            the embedding Tensor).
        start_token (int list or 1D int Tensor, optional): Of shape
            `[batch_size]`. The start tokens.
        end_token (int or int scalar Tensor, optional): The token that marks
            end of decoding.
        **kwargs: additional keyword arguments for constructing the helper.

    Returns:
        An instance of specified helper.
    """
    module_paths = [
        'txtgen.custom',
        'txtgen.modules.decoders.rnn_decoder_helpers',
        'tensorflow.contrib.seq2seq']
    class_kwargs = {"inputs": inputs,
                    "sequence_length": sequence_length,
                    "embedding": embedding,
                    "start_token": start_token,
                    "end_token": end_token}
    class_kwargs.update(kwargs)

    return utils.get_instance_with_redundant_kwargs(
        helper_type, module_paths, class_kwargs)

# TODO (zhiting): to remove
#def _make_training_helper(helper_type,
#                          inputs,
#                          sequence_length,
#                          embedding=None,
#                          **kwargs):
#    """Creates a TrainingHelper instance.
#
#    Args:
#        helper_type (str or class): The type of helper to make. The indicated
#            class must be inherited from
#            :class:`tensorflow.contrib.seq2seq.TrainingHelper`.
#
#            If str, this is the name or full path to the helper class.
#            E.g., the classname of the built-in helpers in
#            :mod:`txtgen.modules.decoders.rnn_decoder_helpers` or
#            :mod:`tensorflow.contrib.seq2seq`, or the classname of user-defined
#            helpers in :mod:`txtgen.custom`, or a full path like
#            "my_module.MyHelper".
#        inputs ((structure of) Tensors): Inputs to the decoder.
#        sequence_length (1D integer list or Tensor): Lengths of input token
#            sequences.
#        embedding (optional): A callable that takes a vector tensor of integer
#            indexes, or the `params` argument for `embedding_lookup` (e.g.,
#            the embedding Tensor).
#        **kwargs: additional keyword arguments for constructing the helper.
#
#    Returns:
#        An instance of specified training helper.
#    """
#    helper_class = helper_type
#    if isinstance(helper_class, str):
#        helper_class = utils.get_class(
#            helper_type,
#            ['txtgen.custom',
#             'txtgen.modules.decoders.rnn_decoder_helpers',
#             'tensorflow.contrib.seq2seq'])
#
#    if not inspect.isclass(helper_class) or \
#            not issubclass(helper_class, TFTrainingHelper):
#        raise ValueError(
#            "If `helper_type` is not a name or full path to a "
#            "helper class, it must be a class inherits `TrainingHelper`.")
#    helper_kwargs = {"inputs": inputs,
#                     "sequence_length": sequence_length}
#    helper_kwargs.update(kwargs)
#    if helper_class != TFTrainingHelper:
#        helper_kwargs.update({"embedding": embedding})
#    return helper_class(**helper_kwargs)


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

