"""
Various neural network layers
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
from txtgen import context
from txtgen.hyperparams import HParams
from txtgen.core.utils import get_instance, switch_dropout

# pylint: disable=not-context-manager, redefined-variable-type, invalid-name

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

def get_rnn_cell_trainable_variables(cell):
    """Returns the list of trainable variables of an RNN cell.

    Args:
        cell: an instance of :class:`tensorflow.contrib.rnn.RNNCell`.

    Returns:
        list: trainable variables of the cell.
    """
    cell_ = cell
    while True:
        try:
            return cell_.trainable_variables
        except AttributeError:
        # Cell wrappers (e.g., `DropoutWrapper`) cannot directly access to
        # `trainable_variables` as they don't initialize superclass
        # (tf==v1.3). So try to access through the cell in the wrapper.
            cell_ = cell._cell

def default_embedding_hparams():
    """Returns default hyperparameters of token embedding used in encoders,
    decoders, and other modules.

    Returns:
        A dictionary with the following structure and values.

        .. code-block:: python

            {
                "name": "embedding",
                "dim": 100,
                "initializer": {
                    "type": "tensorflow.random_uniform_initializer",
                    "kwargs": {
                        "minval": -0.1,
                        "maxval": 0.1,
                        "seed": None
                    }
                },
                "trainable": True,
            }

        Here:

        "name" : str
            Name of the embedding variable.

        "dim" : int
            Embedding dimension.

        "initializer" : dict
            Hyperparameters of the initializer for the embedding values,
            including:

            "type" : str
                Name or full path to the initializer class. The class
                can be

                - Built-in initializer defined in
                  :tf_main:`tf.initializers <initializers>`, e.g.,
                  :tf_main:`tf.initializers.random_uniform
                  <random_uniform_initializer>` (a.k.a
                  tf.random_uniform_initializer) or in
                  :mod:`tensorflow`, e.g.,
                  :tf_main:`tf.glorot_uniform_initializer
                  <glorot_uniform_initializer>`.
                - User-defined initializer in :mod:`txtgen.custom`.
                - External initializer. Must provide the full path, \
                  e.g., :attr:`"my_module.MyInitializer"`.

                The default value is
                :attr:`"tensorflow.random_uniform_initializer"`.

            "kwargs" : dict
                A dictionary of arguments for constructor of the
                initializer class. An initializer is created by
                calling `initialzier_class(**kwargs)` where
                :attr:`initializer_class` is specified in :attr:`"type"`.

                The default value is:

                    .. code-block:: python

                        {
                            "minval": -0.1,
                            "maxval": 0.1,
                            "seed": None
                        }
                which are the arguments of constructing
                :tf_main:`tf.random_uniform_initializer
                <random_uniform_initializer>`.

        "trainable" : bool
            Whether the embedding is trainable.
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
            :func:`~txtgen.core.layers.default_embedding_hparams` for all
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
            initializer = get_instance(
                hparams["initializer"]["type"], kwargs,
                ["txtgen.custom", "tensorflow", "tensorflow.contrib.layers"])
            return tf.get_variable(name=hparams["name"],
                                   shape=[vocab_size, hparams["dim"]],
                                   initializer=initializer,
                                   trainable=hparams["trainable"])
        else:
            return tf.get_variable(name=hparams["name"],
                                   initializer=init_values,
                                   trainable=hparams["trainable"])

def default_conv1d_kwargs():
    """Returns the default keyword argument values of 1D convolution layer(s)
    defined in :tf_main:`tf.layers.Conv1D <layers/Conv1D>`.

    Some of the keyword arguments allow extended values as detailed in the
    following.

    Returns:
        .. code-block:: python

            {
                "kernel_size": [3,4,5],
                "filters": 100,
                "strides": 1,
                "activation": "tensorflow.identity",
                "kernel_initializer": {
                    "type": "tensorflow.glorot_uniform_initializer",
                    "kwargs": {}
                },
                "bias_initializer": {
                    "type": "tensorflow.zeros_initializer",
                    "kwargs": {}
                },
                "kernel_regularizer": None,
                "bias_regularizer": None,
                "activity_regularizer": None
            }

        Here:

        "kernel_size" : int or a list of int
            The length(s) of 1D convolution window(s). If a list, filters with
            different window lengths as specified in the list are created.

            The default value is `[3,4,5]`, which creates 3 sets of filters,
            each of which are with lengths 3, 4, and 5.

        "filters" : int or a list of int
            The number of filters in the convolution. If an int, equal number of
            filters with different window lengths are created. If a list,
            the list must be of the same length as the list in
            :attr:`"kernel_size"`, and each integer in the list is the number
            of filters with respective window length.

            The default value is `100`, which creates 100 filters for each
            filter set.

        "strides" : int or a list of int
            The stride length of the convolution. If an int, the stride length
            is shared across all filter sets. If a list, the list must be of
            the same length as the list in :attr:`"kernel_size"`.

            The default value is `1`.

        "dilation_rate" : int or a list of int
            The dilation rate to use for dilated convolution. If an int, the
            dilation rate is shared across all filter sets. If a list, the list
            must be of the same length as the list in :attr:`"kernel_size"`.

            The default value is `1`.

        "activation" : str
            The name or full path to the activation function applied to the
            outputs of the layer.

            The default value is "tensorflow.identity", which is a linear
            activation.

        "kernel_initializer" : dict
            Hyperparameters of the initializer for the filters, including
            :attr:`"type"` (str) and :attr:`"kwargs"` (dict).

            The default is :tf_main:`tf.glorot_uniform_initializer
            <glorot_uniform_initializer>`.

        "bias_initializer" : dict
            Hyperparameters of the initializer for the bias, including
            :attr:`"type"` (str) and :attr:`"kwargs"` (dict).

            The default is :tf_main:`tf.zeros_initializer <zeros_initializer>`.

        "kernel_regularizer" : dict
            Optional hyperparameters of the regularizer for the convolution
            filters, including :attr:`"type"` (str) and :attr:`"kwargs"` (dict).

            The default value is `None`, i.e., no regularization is performed.

        "bias_regularizer" : dict
            Optional hyperparameters of the regularizer for the bias,
            including :attr:`"type"` (str) and :attr:`"kwargs"` (dict).

            The default value is `None`, i.e., no regularization is performed.

        "activity_regularizer" : dict
            Optional hyperparameters of the regularizer for the layer output,
            including :attr:`"type"` (str) and :attr:`"kwargs"` (dict).

            The default value is `None`, i.e., no regularization is performed.
    """
    return {
        "kernel_size": [3,4,5],
        "filters": 100,
        "strides": 1,
        "dilation_rate": 1,
        "activation": "tensorflow.identity",
        "kernel_initializer": {
            "type": "tensorflow.glorot_uniform_initializer",
            "kwargs": {}
        },
        "bias_initializer": {
            "type": "tensorflow.zeros_initializer",
            "kwargs": {}
        },
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None
    }


def sinusoid_positional_encoding(inputs,
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
    batch_size, max_time, hidden_dim = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        position_idx = tf.tile(tf.expand_dims(tf.range(max_time), 0), [batch_size, 1]) #batch_size * max_time
        position_enc = np.array([
            [pos /np.power(10000, 2.*i/hidden_dim) for i in range(hidden_dim)]
            for pos in range(max_time)])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

        lookup_table = tf.convert_to_tensor(position_enc)
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, hidden_dim]),
                lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_idx)
        if scale:
            outputs = outputs * hidden_dim**0.5
        return outputs

#TODO(zhiting): fix code style
def multihead_attention(queries,
                        keys,
                        num_units= None,
                        num_heads=8,
                        dropout_rate=0,
                        causality = False,
                        scope = 'multihead_attention',
                        reuse= None):
    """perform multihead attention
    Args:
        queries: A 3d tensor with shape of [N, T_q, C_q].
        keys: A 3d tensor with shape of [N, T_k, C_k].
        num_units: A scalar. Attention size.
        dropout_rate: A floating point number.
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
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=context.is_train())

        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize

    return outputs



def poswise_feedforward(attended_dec, scope="multihead_attention", reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    hidden_dim = attended_dec.shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.conv1d(inputs = attended_dec,
                filters=hidden_dim*4,
                kernel_size=1,
                activation=tf.nn.relu,
                use_bias=True)
        outputs = tf.layers.conv1d(inputs = outputs,
                filters=hidden_dim,
                kernel_size=1,
                activation=None,
                use_bias=True)
        outputs += attended_dec #residual connection
    return outputs

def normalize(inputs,
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta

    return outputs

