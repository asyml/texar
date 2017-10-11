#
"""
Various neural network layers
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from txtgen.hyperparams import HParams
from txtgen.core.utils import get_instance, switch_dropout

# pylint: disable=not-context-manager, redefined-variable-type

def default_rnn_cell_hparams():
    """Returns default hyperparameters of an RNN cell.

    Returns:
        dict: A dictionary with the following structure and values:

        .. code-block:: python

            {
                "cell": {
                    # Name or full path of the cell class. E.g., the classname
                    # of built-in cells in `tensorflow.contrib.rnn`, or the
                    # classname of user-defined cells in `txtgen.custom`, or a
                    # full path like "my_module.MyCell".
                    "type": "BasicLSTMCell",

                    # A dictionary of arguments for constructor of the cell
                    # class. An RNN cell is created by calling the cell class
                    # named in `type` passing the arguments specified in
                    # `kwargs` as `cell_class(**kwargs)`
                    "kwargs": {
                        "num_units": 64
                    }
                },

                # Number of cell layers
                "num_layers": 1

                # Dropout applied to the cell in each layer. See
                # `tensorflow.contrib.rnn.DropoutWrapper` for each of the
                # hyperparameters. If all keep probablities are 1.0, no dropout
                # is applied.
                "dropout": {
                    "input_keep_prob": 1.0,
                    "output_keep_prob": 1.0,
                    "state_keep_prob": 1.0,

                    # If True, the same dropout mask is applied at every step,
                    # and the list of input size of each layer is required
                    # (in "input_size"). The input size of a layer is the size
                    # of the last dimension of its input tensor. E.g., the
                    # input size of the first layer is usually the dimension of
                    # word embeddings, while the input size of followup layers
                    # are usually the num_units of the cells.
                    "variational_recurrent": False,
                    "input_size": []
                },

                # If True, apply residual connection on the inputs and
                # outputs of cell in each layer except the first layer.
                "residual": False,

                # If True, apply highway connection on the inputs and
                # outputs of cell in each layer except the first layer.
                "highway": False,
            }
    """
    return {
        "cell": {
            "type": "BasicLSTMCell",
            "kwargs": {
                "num_units": 64
            }
        },
        "num_layers": 1,
        "dropout": {
            "input_keep_prob": 1.0,
            "output_keep_prob": 1.0,
            "state_keep_prob": 1.0,
            "variational_recurrent": False,
            "input_size": []
        },
        "residual": False,
        "highway": False,
    }


def get_rnn_cell(hparams=None):
    """Creates an RNN cell.

    See :meth:`~txtgen.core.layers.default_rnn_cell_hparams` for all
    hyperparameters and default values.

    Args:
        hparams (dict or HParams, optional): Cell hyperparameters. Missing
            hyperparameters are set to default values.

    Returns:
        An instance of `RNNCell`.
    """
    if hparams is None or isinstance(hparams, dict):
        hparams = HParams(hparams, default_rnn_cell_hparams())

    d_hp = hparams["dropout"]
    if d_hp["variational_recurrent"] and \
            len(d_hp["input_size"]) != hparams["num_layers"]:
        raise ValueError(
            "If variational_recurrent=True, input_size must be a list of "
            "num_layers(%d) integers. Got len(input_size)=%d." %
            (hparams["num_layers"], len(d_hp["input_size"])))

    cells = []
    cell_kwargs = hparams["cell"]["kwargs"].todict()
    for layer_i in range(hparams["num_layers"]):
        # Create the basic cell
        cell_type = hparams["cell"]["type"]
        cell_modules = ['txtgen.custom', 'tensorflow.contrib.rnn']
        cell = get_instance(cell_type, cell_kwargs, cell_modules)

        # Optionally add dropout
        if d_hp["input_keep_prob"] < 1.0 or \
                d_hp["output_keep_prob"] < 1.0 or \
                d_hp["state_keep_prob"] < 1.0:
            vr_kwargs = {}
            if d_hp["variational_recurrent"]:
                vr_kwargs = {"variational_recurrent": True,
                             "input_size": d_hp["input_size"][layer_i],
                             "dtype": tf.float32}
            cell = rnn.DropoutWrapper(
                cell=cell,
                input_keep_prob=switch_dropout(d_hp["input_keep_prob"]),
                output_keep_prob=switch_dropout(d_hp["output_keep_prob"]),
                state_keep_prob=switch_dropout(d_hp["state_keep_prob"]),
                **vr_kwargs)

        # Optionally add residual and highway connections
        if layer_i > 0:
            if hparams["residual"]:
                cell = rnn.ResidualWrapper(cell)
            if hparams["highway"]:
                cell = rnn.HighwayWrapper(cell)

        cells.append(cell)

    if hparams["num_layers"] > 1:
        cell = rnn.MultiRNNCell(cells)
    else:
        cell = cells[0]

    return cell


def default_embedding_hparams():
    """Returns default hyperparameters of embedding used in modules.

    Returns:
        dict: A dictionary with the following structure and values:

        .. code-block:: python

            {
                "name": "embedding", # A string. Name of the embedding variable.
                "dim": 100,          # An integer. Embedding dimension.
                "initializer": {     # Initializer of embedding values.
                    # A string. Name or full path to the initializer class.
                    # An initializer is a class inheriting from
                    # `tensorflow.Initializer`, which can be built-in
                    # classes in module `tensorflow`, or user-defined
                    # classes in `txtgen.custom`, or a full path like
                    # `my_module.MyInitializer`.
                    "type": "tensorflow.random_uniform_initializer",

                    # A dictionary of arguments for constructor of the
                    # initializer class. An initializer is created by
                    # calling `initialzier_class(**kwargs)` where
                    # `initializer_class` is specified in `type`.
                    "kwargs": {
                        "minval": -0.1,
                        "maxval": 0.1,
                        "seed": None
                    }
                },
                # (bool) Whether the embedding variable trainable.
                "trainable": True,
            }
    """
    return { #TODO(zhiting): allow more hparams like regularizer
        "name": "embedding",
        "dim": 50,
        "initializer": {
            "type": "tensorflow.random_uniform_initializer",
            "kwargs": {
                "minval": -0.1,
                "maxval": 0.1,
                "seed": None
            }
        },
        "trainable": True
    }


def get_embedding(hparams=None,
                  init_values=None,
                  vocab_size=None,
                  variable_scope=None):
    """Creates embedding variable if not exists.

    Args:
        hparams (dict or HParams, optional): Embedding hyperparameters. Missing
            hyperparameters are set to default values. See
            :meth:`~txtgen.core.layers.default_embedding_hparams` for all
            hyperparameters and default values.

            If :attr:`init_values` is given, :attr:`hparams["initializer"]`,
            and :attr:`hparams["dim"]` are ignored.
        init_values (Tensor or numpy array, optional): Initial values of the
            embedding variable. If not given, embedding is initialized as
            specified in :attr:`hparams["initializer"]`.
        vocab_size (int, optional): The vocabulary size. Required if
            :attr:`init_values` is not provided.
        variable_scope (string or VariableScope, optional): Variable scope of
            the embedding variable.

    Returns:
        Variable: A 2D `Variable` of the same shape with :attr:`init_values`
        or of the shape :attr:`[vocab_size, hparams["dim"]]`.
    """
    with tf.variable_scope(variable_scope, "embedding"):
        if hparams is None or isinstance(hparams, dict):
            hparams = HParams(hparams, default_embedding_hparams())
        if init_values is None:
            kwargs = hparams["initializer"]["kwargs"].todict()
            initializer = get_instance(hparams["initializer"]["type"],
                                       kwargs,
                                       ["txtgen.custom", "tensorflow"])
            return tf.get_variable(name=hparams["name"],
                                   shape=[vocab_size, hparams["dim"]],
                                   initializer=initializer,
                                   trainable=hparams["trainable"])
        else:
            return tf.get_variable(name=hparams["name"],
                                   initializer=init_values,
                                   trainable=hparams["trainable"])

def sinuoid_positional_encoding(inputs,
                                zero_pad=True,
                                scale=True,
                                reuse=None,
                                position_duration=10000,
                                scope='sinuoid_positional_embedding'):
    """obtain a positional encoding of inputs
    Args:
        inputs: [Tensor] A Tensor of shape `[batch_size, max_time, hidden_dim]`
        max_time: [Int], max time steps
        hidden_dim: [Int], hidden size of embedding
        zero_pad: [Boolean], If True, all the values of the first row(id = 0) should be constant zero
        scale: [Boolean], If True, the output will be multiplied by sqrt num_units(check details from paper)
        scope: [String], Optional scope for 'variable_scope'
        position_duration: [Int], default=10000
    """
    with tf.variable_scope(scope, reuse=reuse):
        batch_size, max_time, hidden_dim = inputs.get_shape().as_list()
        input_one = tf.tile(tf.expand_dims(tf.range(max_time), 0), [batch_size, 1]) #batch_size * max_time
        position_block = tf.tile(tf.expand_dims(tf.range(max_time), 1), [1, num_units // 2])
        unit_block = tf.tile(tf.expand_dims(tf.range(hidden_dim // 2), 0), [max_time, 1])
        rad_block = tf.pow(tf.div(position_block, tf.multiply(position_duration, 1)), tf.div(unit_block, hidden_dim // 2))

        sin_block = tf.sin(tf.cast(rad_block, tf.float32))
        cos_block = tf.cos(tf.cast(rad_block, tf.float32))
        lookup_table = tf.concat([sin_block, cos_block], axis = 1)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape = [1, num_units]), lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, input_one)
        if scale:
            outputs = outputs * math.sqrt(hidden_dim)
        return outputs


def multihead_attention(queries,
                        keys,
                        num_units= None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality = False,
                        scope = 'multihead_attention',
                        reuse= None):
    """perform multihead attention
    Args:
        queries: A 3d tensor with shape of [N, T_q, C_q].
        keys: A 3d tensor with shape of [N, T_k, C_k].
        num_units: A scalar. Attention size.
        dropout_rate: A floating point number.
        is_training: Boolean. Controller of mechanism for dropout.
        causality: Boolean. Should be true, units that reference the future are masked
        num_heads: An int. Number of heads.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    Returns
        A 3d tensor with shape of (N, T_q, C)
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)

        # According to the paper, there is a scale operation
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)  The upper triangle of the last two dimensions is ignored.
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)

    return outputs
