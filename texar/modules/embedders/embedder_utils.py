#
"""Utils of embedder.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from texar.hyperparams import HParams
from texar.core import layers

__all__ = [
    "default_embedding_hparams",
    "get_embedding",
]

def default_embedding_hparams():
    """Returns default hyperparameters of token embedding used in encoders,
    decoders, and other modules.

    Returns:
        A dictionary with the following structure and values.

        .. code-block:: python

            {
                "name": "embedding",
                "dim": 100,
                "initializer": None,
                "regularizer": {
                    "type": "L1L2",
                    "kwargs": {
                        "l1": 0.,
                        "l2": 0.
                    }
                },
                "dropout_rate": 0,
                "trainable": True,
            }

        Here:

        "name" : str
            Name of the embedding variable.

        "dim" : int
            Embedding dimension.

        "initializer" : dict or None
            Hyperparameters of the initializer for the embedding values. An
            example is as

            .. code-block:: python

                {
                    "type": "random_uniform_initializer",
                    "kwargs": {
                        "minval": -0.1,
                        "maxval": 0.1,
                        "seed": None
                    }
                }

            which corresponds to :tf_main:`tf.random_uniform_initializer
            <random_uniform_initializer>`, and includes:

            "type" : str or initializer instance
                Name, full path, or instance of the initializer class; Or name
                or full path to a function that returns the initializer class.
                The class or function can be

                - Built-in initializer defined in \
                  :tf_main:`tf.initializers <initializers>`, e.g., \
                  :tf_main:`random_uniform <random_uniform_initializer>` \
                  (a.k.a :class:`tf.random_uniform_initializer`), or \
                  in :mod:`tf`, e.g., :tf_main:`glorot_uniform_initializer \
                  <glorot_uniform_initializer>`, or in \
                  :tf_main:`tf.keras.initializers <keras/initializers>`.
                - User-defined initializer in :mod:`texar.custom`.
                - External initializer. Must provide the full path, \
                  e.g., :attr:`"my_module.MyInitializer"`, or the instance.

            "kwargs" : dict
                A dictionary of arguments for constructor of the
                initializer class or for the function. An initializer is
                created by `initialzier = initializer_class_or_fn(**kwargs)`
                where :attr:`initializer_class_or_fn` is specified in
                :attr:`"type"`.
                Ignored if :attr:`"type"` is an initializer instance.

        "regularizer" : dict
            Hyperparameters of the regularizer for the embedding values. The
            regularizer must be an instance of
            the base :tf_main:`Regularizer <keras/regularizers/Regularizer>`
            class. The hyperparameters include:

            "type" : str or Regularizer instance
                Name, full path, or instance of the regularizer class. The
                class can be

                - Built-in regularizer defined in
                  :tf_main:`tf.keras.regularizers <keras/regularizers>`, e.g.,
                  :tf_main:`L1L2 <keras/regularizers/L1L2>`.
                - User-defined regularizer in :mod:`texar.custom`. The
                  regularizer class should inherit the base class
                  :tf_main:`Regularizer <keras/regularizers/Regularizer>`.
                - External regularizer. Must provide the full path, \
                  e.g., :attr:`"my_module.MyRegularizer"`, or the instance.

            "kwargs" : dict
                A dictionary of arguments for constructor of the
                regularizer class. A regularizer is created by
                calling `regularizer_class(**kwargs)` where
                :attr:`regularizer_class` is specified in :attr:`"type"`.
                Ignored if :attr:`"type"` is a Regularizer instance.

            The default value corresponds to
            :tf_main:`L1L2 <keras/regularizers/L1L2>` with `(l1=0, l2=0)`,
            which disables regularization.

        "dropout_rate" : float
            The dropout rate between 0 and 1. E.g., `dropout_rate=0.1` would
            drop out 10% of the embedding.

        "trainable" : bool
            Whether the embedding is trainable.
    """
    return {
        "name": "embedding",
        "dim": 100,
        "initializer": None,
        "regularizer": layers.default_regularizer_hparams(),
        "dropout_rate": 0,
        "trainable": True
    }


def get_embedding(hparams=None,
                  init_value=None,
                  vocab_size=None,
                  variable_scope='Embedding'):
    """Creates embedding variable if not exists.

    Args:
        hparams (dict or HParams, optional): Embedding hyperparameters. Missing
            hyperparameters are set to default values. See
            :func:`~texar.modules.default_embedding_hparams`
            for all hyperparameters and default values.

            If :attr:`init_value` is given, :attr:`hparams["initializer"]`,
            and :attr:`hparams["dim"]` are ignored.
        init_value (Tensor or numpy array, optional): Initial values of the
            embedding variable. If not given, embedding is initialized as
            specified in :attr:`hparams["initializer"]`.
        vocab_size (int, optional): The vocabulary size. Required if
            :attr:`init_value` is not provided.
        variable_scope (str or VariableScope, optional): Variable scope of
            the embedding variable.

    Returns:
        Variable or Tensor: A 2D `Variable` or `Tensor` of the same shape with
        :attr:`init_value` or of the shape
        :attr:`[vocab_size, hparams["dim"]]`.
    """
    with tf.variable_scope(variable_scope):
        if hparams is None or isinstance(hparams, dict):
            hparams = HParams(hparams, default_embedding_hparams())
        regularizer = layers.get_regularizer(hparams["regularizer"])
        if init_value is None:
            initializer = layers.get_initializer(hparams["initializer"])
            embedding = tf.get_variable(name=hparams["name"],
                                        shape=[vocab_size, hparams["dim"]],
                                        initializer=initializer,
                                        regularizer=regularizer,
                                        trainable=hparams["trainable"])
        else:
            embedding = tf.get_variable(name=hparams["name"],
                                        initializer=tf.to_float(init_value),
                                        regularizer=regularizer,
                                        trainable=hparams["trainable"])

        #if hparams["dropout_rate"] > 0.:
        #    keep_prob = utils.switch_dropout(
        #        hparams["dropout"]["keep_prob"], mode)
        #    embedding = tf.nn.dropout(embedding, keep_prob=keep_prob)
        #    # TODO: Return value type changed and may not be compatible with
        #    # previous semantic.
        return embedding
