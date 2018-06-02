#
"""
Various RNN encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
from tensorflow.contrib.framework import nest

from texar.modules.encoders.encoder_base import EncoderBase
from texar.modules.networks.conv_networks import _to_list
from texar.core import layers
from texar.utils import utils
from texar.utils.shapes import mask_sequences
from texar.hyperparams import HParams

# pylint: disable=too-many-arguments, too-many-locals, invalid-name, no-member

__all__ = [
    "_forward_single_output_layer",
    "RNNEncoderBase",
    "UnidirectionalRNNEncoder",
    "BidirectionalRNNEncoder"
]

def _default_output_layer_hparams():
    return {
        "num_layers": 0,
        "layer_size": 128,
        "activation": "identity",
        "final_layer_activation": None,
        "other_dense_kwargs": None,
        "dropout_layer_ids": [],
        "dropout_rate": 0.5,
        "variational_dropout": False,
        "@no_typecheck": ["activation", "final_layer_activation",
                          "layer_size", "dropout_layer_ids"]
    }

def _build_dense_output_layer(hparams):
    nlayers = hparams.num_layers

    if nlayers <= 0:
        return None

    layer_size = _to_list(
        hparams.layer_size, 'output_layer.layer_size', nlayers)

    other_kwargs = hparams.other_dense_kwargs or {}
    if isinstance(other_kwargs, HParams):
        other_kwargs = other_kwargs.todict()
    if not isinstance(other_kwargs, dict):
        raise ValueError(
            "hparams 'output_layer.other_dense_kwargs' must be a dict.")

    dense_layers = []
    for i in range(nlayers):
        if i == nlayers - 1:
            activation = hparams.final_layer_activation
        else:
            activation = hparams.activation

        kwargs_i = {"units": layer_size[i],
                    "activation": activation,
                    "name": "dense_%d" % (i+1)}
        kwargs_i.update(other_kwargs)

        layer_hparams = {"type": "Dense", "kwargs": kwargs_i}
        dense_layers.append(layers.get_layer(hparams=layer_hparams))

    if len(dense_layers) == 1:
        dense_layers = dense_layers[0]

    return dense_layers

def _forward_single_output_layer(inputs, output_layer, flatten_inputs=True):
    """Forwards the input through a single output layer.

    :attr:`inputs` is a Tensor of shape `[batch_size, max_time, dim]`
    or `[max_time, batch_size, dim]`.
    """
    d3_shape = tf.concat([tf.shape(inputs)[:2], [-1]], axis=0)
    # Reshape inputs to [-1, dim]
    if flatten_inputs:
        inputs = tf.reshape(inputs, d3_shape)
    inputs_T = tf.transpose(inputs, perm=[2, 0, 1])
    inputs_flat = tf.transpose(tf.layers.flatten(inputs_T), perm=[1, 0])
    # Feed to the layer
    output_flat = output_layer(inputs_flat)
    # Reshape output to [batch_size/max_time, max_time/batch_size, new_dim]
    output = tf.reshape(output_flat, d3_shape)
    return output

def _apply_dropout(inputs, time_major, hparams, training):
    """Applies dropout to the inputs.

    :attr:`inputs` is a Tensor of shape `[batch_size, max_time, dim]`
    if :attr:`time_major=False`, or shape `[max_time, batch_size, dim]`
    if :attr:`time_major=True`.
    """
    noise_shape = None
    if hparams.variational_dropout:
        if time_major:
            noise_shape = [1, None, None]
        else:
            noise_shape = [None, 1, None]
    return tf.layers.dropout(inputs, rate=hparams.dropout_rate,
                             noise_shape=noise_shape, training=training)

def _forward_output_layers(inputs, output_layer, time_major, hparams, mode,
                           sequence_length=None):
    """Forwards inputs through the output layers.

    :attr:`inputs` is a Tensor of shape `[batch_size, max_time, dim]`
    if :attr:`time_major=False`, or shape `[max_time, batch_size, dim]`
    if :attr:`time_major=True`.

    Returns a Tensor of shape `[batch_size, max_time, new_dim]` or
    `[max_time, batch_size, new_dim]`.
    """
    if output_layer is None:
        return inputs

    if hparams is None:
        # output_layer was passed in from the constructor
        if isinstance(output_layer, (list, tuple)):
            raise ValueError('output_layer must not be a list or tuple.')
        output = _forward_single_output_layer(inputs, output_layer)
    else:
        # output_layer was built based on hparams
        output_layer = _to_list(output_layer)

        dropout_layer_ids = _to_list(hparams.dropout_layer_ids)
        if len(dropout_layer_ids) > 0:
            training = utils.is_train_mode(mode)

        output = inputs
        for i, layer in enumerate(output_layer):
            if i in dropout_layer_ids:
                output = _apply_dropout(output, time_major, hparams, training)
            output = _forward_single_output_layer(output, layer)

        if len(output_layer) in dropout_layer_ids:
            output = _apply_dropout(output, time_major, hparams, training)

    if sequence_length is not None:
        output = mask_sequences(
            output, sequence_length, time_major=time_major, tensor_rank=3)

    return output

class RNNEncoderBase(EncoderBase):
    """Base class for all RNN encoder classes.

    Args:
        hparams (dict, optional): Encoder hyperparameters. If it is not
            specified, the default hyperparameter setting is used. See
            :attr:`default_hparams` for the sturcture and default values.
    """

    def __init__(self, hparams=None):
        EncoderBase.__init__(self, hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            .. code-block:: python

                {
                    "name": "rnn_encoder"
                }

            Here:

            "name" : str
                Name of the encoder.
        """
        return {
            "name": "rnn_encoder"
        }

    def _build(self, inputs, *args, **kwargs):
        """Encodes the inputs.

        Args:
          inputs: Inputs to the encoder.
          *args: Other arguments.
          **kwargs: Keyword arguments.

        Returns:
          Encoding results.
        """
        raise NotImplementedError


class UnidirectionalRNNEncoder(RNNEncoderBase):
    """One directional RNN encoder.

    Args:
        cell: (RNNCell, optional) If it is not specified,
            a cell is created as specified in :attr:`hparams["rnn_cell"]`.
        cell_dropout_mode (optional): A Tensor taking value of
            :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, which
            toggles dropout in the RNN cell (e.g., activates dropout in the
            TRAIN mode). If `None`, :func:`~texar.context.global_mode` is used.
            Ignored if :attr:`cell` is given.
        output_layer (optional): An instance of
            :tf_main:`tf.layers.Layer <layers/Layer>`. Apply to the RNN cell
            output of each step. If `None` (default), the output layer is
            created as specified in :attr:`hparams["output_layer"]`.
        hparams (dict, optional): Encoder hyperparameters. If it is not
            specified, the default hyperparameter setting is used. See
            :attr:`default_hparams` for the sturcture and default values.
            Missing values will take default.
    """

    def __init__(self,
                 cell=None,
                 cell_dropout_mode=None,
                 output_layer=None,
                 hparams=None):
        RNNEncoderBase.__init__(self, hparams)

        # Make RNN cell
        with tf.variable_scope(self.variable_scope):
            if cell is not None:
                self._cell = cell
            else:
                self._cell = layers.get_rnn_cell(
                    self._hparams.rnn_cell, cell_dropout_mode)

        # Make output layer
        with tf.variable_scope(self.variable_scope):
            if output_layer is not None:
                self._output_layer = output_layer
                self._output_layer_hparams = None
            else:
                self._output_layer = _build_dense_output_layer(
                    self._hparams.output_layer)
                self._output_layer_hparams = self._hparams.output_layer

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            .. code-block:: python

                {
                    "rnn_cell": default_rnn_cell_hparams(),
                    "output_layer": {
                        "num_layers": 0,
                        "layer_size": 128,
                        "activation": "identity",
                        "final_layer_activation": None,
                        "other_dense_kwargs": None,
                        "dropout_layer_ids": [],
                        "dropout_rate": 0.5,
                        "variational_dropout": False
                    },
                    "name": "unidirectional_rnn_encoder"
                }

            Here:

            "rnn_cell" : dict
                A dictionary of RNN cell hyperparameters. Ignored if
                :attr:`cell` is given when constructing the encoder.

                The default value is defined in
                :func:`~texar.core.layers.default_rnn_cell_hparams`.

            "output_layer" : dict
                Output layer hyperparameters. Ignored if :attr:`output_layer`
                is given in the constructor. Includes:

                "num_layers" : int
                    The number of output (dense) layers. Set to 0 to avoid any
                    output layers applied to the cell outputs..

                "layer_size" : int or list
                    The size of each of the output (dense) layers.

                    If an `int`, each output layer will have the same size. If
                    a list, the length must equal to :attr:`num_layers`.

                "activation" : str or callable or None
                    The activation function for each of the output (dense)
                    layer except for the final layer. This can be
                    the function itself, or its string name or full path.

                    E.g., `"activation": tensorflow.nn.relu`
                    or `"activation": "relu"`
                    or `"activation": "tensorflow.nn.relu"`

                    Default is `None` which maintains a linear activation.

                "final_layer_activation" : str or callable or None
                    The activation function for the final output layer.

                "other_dense_kwargs" : dict or None
                    Other keyword arguments to construct each of the output
                    dense layers, e.g., :attr:`use_bias`. See
                    :tf_main:`Dense <layers/Dense>` for the arguments.

                    E.g., `"other_dense_kwargs": { "use_bias": False }`.

                "dropout_layer_ids" : int or list
                    The indexes of layers (starting from `0`) whose inputs
                    are applied with dropout. The index = :attr:`num_layers`
                    means dropout applies to the final layer output. E.g.,

                    .. code-block:: python

                        {
                            "num_layers": 2,
                            "dropout_layer_ids": [0, 2]
                        }

                    will leads to a series of layers as
                    `-dropout-layer0-layer1-dropout-`.

                    The dropout mode (training or not) is controlled
                    by the :attr:`mode` argument when calling the encoder.

                "dropout_rate" : float
                    The dropout rate, between 0 and 1. E.g.,
                    `"dropout_rate": 0.1` would drop out 10% of elements.

                "variational_dropout": bool
                    Whether the dropout mask is the same across all time steps.

            "name" : str
                Name of the encoder
        """
        hparams = RNNEncoderBase.default_hparams()
        hparams.update({
            "rnn_cell": layers.default_rnn_cell_hparams(),
            "output_layer": _default_output_layer_hparams(),
            "name": "unidirectional_rnn_encoder"
        })
        return hparams

    def _build(self,
               inputs,
               sequence_length=None,
               initial_state=None,
               time_major=False,
               mode=None,
               return_cell_output=False,
               **kwargs):
        """Encodes the inputs.

        Args:
            inputs: A 3D Tensor of shape `[batch_size, max_time, dim]`.
                The first two dimensions
                :attr:`batch_size` and :attr:`max_time` are exchanged if
                :attr:`time_major=True` is specified.
            sequence_length (int list or 1D Tensor, optional): Sequence lengths
                of the batch inputs. Used to copy-through state and zero-out
                outputs when past a batch element's sequence length.
            initial_state (optional): Initial state of the RNN.
            time_major (bool): The shape format of the :attr:`inputs` and
                :attr:`outputs` Tensors. If `True`, these tensors are of shape
                `[max_time, batch_size, depth]`. If `False` (default),
                these tensors are of shape `[batch_size, max_time, depth]`.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, including
                `TRAIN`, `EVAL`, and `PREDICT`. Controls output layer dropout
                if the output layer is specified with :attr:`hparams`.
                If `None` (default), :func:`texar.context.global_mode()`
                is used.
            return_cell_output (bool): Whether to return the output of the RNN
                cell. This is the results prior to the output layer.
            **kwargs: Optional keyword arguments of
                :tf_main:`tf.nn.dynamic_rnn <nn/dynamic_rnn>`,
                such as `swap_memory`, `dtype`, `parallel_iterations`, etc.

        Returns:
            If :attr:`return_cell_output` is `False` (default), returns a
            pair :attr:`(outputs, final_state)` where

            - :attr:`outputs`: The RNN output tensor by the output layer \
              (if exists) or the RNN cell (otherwise). The tensor is of shape \
              `[batch_size, max_time, output_dim]` (if \
              :attr:`time_major` == `False`) or \
              `[max_time, batch_size, output_dim]` (if \
              :attr:`time_major` == `True`). \

              If RNN cell output is a (nested) tuple of Tensors, then the \
              :attr:`outputs` will be a (nested) tuple having the same \
              structure as the cell output.

            - :attr:`final_state`: The final state of the RNN, which is a \
              Tensor of shape `[batch_size] + cell.state_size` or \
              a (nested) tuple of Tensors (if `cell.state_size` is a (nested) \
              tuple).

            If :attr:`return_cell_output` is `True`, returns a triple
            :attr:`(outputs, final_state, cell_outputs)` where

            - :attr:`cell_outputs`: The outputs by the RNN cell prior to the \
              output layer, having the same structure with :attr:`outputs` \
              except for the `output_dim`.
        """
        if ('dtype' not in kwargs) and (initial_state is None):
            cell_outputs, state = tf.nn.dynamic_rnn(
                cell=self._cell,
                inputs=inputs,
                sequence_length=sequence_length,
                initial_state=initial_state,
                time_major=time_major,
                dtype=tf.float32,
                **kwargs)
        else:
            cell_outputs, state = tf.nn.dynamic_rnn(
                cell=self._cell,
                inputs=inputs,
                sequence_length=sequence_length,
                initial_state=initial_state,
                time_major=time_major,
                **kwargs)

        map_func = functools.partial(
            _forward_output_layers,
            output_layer=self._output_layer,
            time_major=time_major,
            hparams=self._output_layer_hparams,
            mode=mode)
        outputs = nest.map_structure(map_func, cell_outputs)

        if not self._built:
            self._add_internal_trainable_variables()
            # Add trainable variables of `self._cell` and `self._output_layer`
            # which may be constructed externally.
            self._add_trainable_variable(
                layers.get_rnn_cell_trainable_variables(self._cell))
            if self._output_layer and \
                    not isinstance(self._output_layer, (list, tuple)):
                self._add_trainable_variable(
                    self._output_layer.trainable_variables)
            self._built = True

        if return_cell_output:
            return outputs, state, cell_outputs
        else:
            return outputs, state

    #def append_layer(self, layer):
    #    """Appends a layer to the end of the output layer. The layer must take
    #    as inputs a 2D Tensor and output another 2D Tensor (e.g., a
    #    :tf_main:`Dense <layers/Dense>` layer).

    #    The method is only feasible before :meth:`_build` is called.

    #    Args:
    #        layer: A :tf_main:`tf.layers.Layer <layers/Layer>` instance, or
    #            a `dict` of layer hyperparameters.
    #    """
    #    if self._built:
    #        raise TexarError("`UnidirectionalRNNEncoder.append_layer` can be "
    #                         "called only before `_build` is called.")

    #    with tf.variable_scope(self.variable_scope):
    #        layer_ = layer
    #        if not isinstance(layer_, tf.layers.Layer):
    #            layer_ = layers.get_layer(hparams=layer_)
    #        if self._output_layer is None:
    #            self._output_layer = layer_
    #        else:
    #            self._output_layer = _to_list(self._output_layer)
    #            self._output_layers.append(layer_)

    @property
    def cell(self):
        """The RNN cell.
        """
        return self._cell

    @property
    def state_size(self):
        """The state size of encoder cell.

        Same as :attr:`encoder.cell.state_size`.
        """
        return self.cell.state_size

    @property
    def output_layer(self):
        """The output layer.
        """
        return self._output_layer

class BidirectionalRNNEncoder(RNNEncoderBase):
    """Bidirectional forward-backward RNN encoder.

    Args:
        cell_fw (RNNCell, optional): The forward RNN cell. If not given,
            a cell is created as specified in :attr:`hparams["rnn_cell_fw"]`.
        cell_bw (RNNCell, optional): The backward RNN cell. If not given,
            a cell is created as specified in :attr:`hparams["rnn_cell_bw"]`.
        cell_dropout_mode (optional): A tensor taking value of
            :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, which
            toggles dropout in the RNN cells (e.g., activates dropout in the
            TRAIN mode). If `None`, :func:`~texar.context.global_mode()` is
            used. Ignored if respective cell is given.
        output_layer_fw (optional): An instance of
            :tf_main:`tf.layers.Layer <layers/Layer>`. Apply to the forward
            RNN cell output of each step. If `None` (default), the output
            layer is created as specified in :attr:`hparams["output_layer_fw"]`.
        output_layer_bw (optional): An instance of
            :tf_main:`tf.layers.Layer <layers/Layer>`. Apply to the backward
            RNN cell output of each step. If `None` (default), the output
            layer is created as specified in :attr:`hparams["output_layer_bw"]`.
        hparams (dict, optional): Encoder hyperparameters. If it is not
            specified, the default hyperparameter setting is used. See
            :attr:`default_hparams` for the sturcture and default values.
            Missing values will take default.
    """

    def __init__(self,
                 cell_fw=None,
                 cell_bw=None,
                 cell_dropout_mode=None,
                 output_layer_fw=None,
                 output_layer_bw=None,
                 hparams=None):
        RNNEncoderBase.__init__(self, hparams)

        # Make RNN cells
        with tf.variable_scope(self.variable_scope):
            if cell_fw is not None:
                self._cell_fw = cell_fw
            else:
                self._cell_fw = layers.get_rnn_cell(
                    self._hparams.rnn_cell_fw, cell_dropout_mode)

            if cell_bw is not None:
                self._cell_bw = cell_bw
            elif self._hparams.rnn_cell_share_config:
                self._cell_bw = layers.get_rnn_cell(
                    self._hparams.rnn_cell_fw, cell_dropout_mode)
            else:
                self._cell_bw = layers.get_rnn_cell(
                    self._hparams.rnn_cell_bw, cell_dropout_mode)

        # Make output layers
        with tf.variable_scope(self.variable_scope):
            if output_layer_fw is not None:
                self._output_layer_fw = output_layer_fw
            else:
                self._output_layer_fw = _build_dense_output_layer(
                    self._hparams.output_layer_fw)

            if output_layer_bw is not None:
                self._output_layer_bw = output_layer_bw
            elif self._hparams.output_layer_share_config:
                self._output_layer_bw = _build_dense_output_layer(
                    self._hparams.output_layer_fw)
            else:
                self._output_layer_bw = _build_dense_output_layer(
                    self._hparams.output_layer_bw)


    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            .. code-block:: python

                {
                    "rnn_cell_fw": default_rnn_cell_hparams(),
                    "rnn_cell_bw": default_rnn_cell_hparams(),
                    "rnn_cell_share_config": True,
                    "output_layer_fw": {
                        "num_layers": 0,
                        "layer_size": 128,
                        "activation": "identity",
                        "final_layer_activation": None,
                        "other_dense_kwargs": None,
                        "dropout_layer_ids": [],
                        "dropout_rate": 0.5,
                        "variational_dropout": False
                    },
                    "output_layer_bw": {
                        # Same as "output_layer_fw"
                        # ...
                    },
                    "output_layer_share_config": True,
                    "name": "bidirectional_rnn_encoder"
                }

            Here:

            "rnn_cell_fw" : dict
                Hyperparameters of the forward RNN cell.
                Ignored if :attr:`cell_fw` is given when constructing
                the encoder.

                The default value is defined in
                :meth:`~texar.core.layers.default_rnn_cell_hparams`.

            "rnn_cell_bw" : dict
                Hyperparameters of the backward RNN cell.
                Ignored if :attr:`cell_bw` is given when constructing
                the encoder, or if :attr:`"rnn_cell_share_config"` is `True`.

                The default value is defined in
                :meth:`~texar.core.layers.default_rnn_cell_hparams`.

            "rnn_cell_share_config" : bool
                Whether share hyperparameters of the backward cell with the
                forward cell. Note that the cell parameters are not shared.

                If `True` (default), :attr:`"rnn_cell_bw"` is ignored.

            "output_layer_fw" : dict
                Hyperparameters of the forward output layer. Ignored if
                :attr:`output_layer_fw` is given in the constructor. Includes:

                "num_layers" : int
                    The number of output (dense) layers. Set to 0 to avoid any
                    output layers applied to the cell outputs..

                "layer_size" : int or list
                    The size of each of the output (dense) layers.

                    If an `int`, each output layer will have the same size. If
                    a list, the length must equal to :attr:`num_layers`.

                "activation" : str or callable or None
                    The activation function for each of the output (dense)
                    layer except for the final layer. This can be
                    the function itself, or its string name or full path.

                    E.g., `"activation": tensorflow.nn.relu`
                    or `"activation": "relu"`
                    or `"activation": "tensorflow.nn.relu"`

                    Default is `None` which maintains a linear activation.

                "final_layer_activation" : str or callable or None
                    The activation function for the final output layer.

                "other_dense_kwargs" : dict or None
                    Other keyword arguments to construct each of the output
                    dense layers, e.g., :attr:`use_bias`. See
                    :tf_main:`Dense <layers/Dense>` for the arguments.

                    E.g., `"other_dense_kwargs": { "use_bias": False }`.

                "dropout_layer_ids" : int or list
                    The indexes of layers (starting from `0`) whose inputs
                    are applied with dropout. The index = :attr:`num_layers`
                    means dropout applies to the final layer output. E.g.,

                    .. code-block:: python

                        {
                            "num_layers": 2,
                            "dropout_layer_ids": [0, 2]
                        }

                    will leads to a series of layers as
                    `-dropout-layer0-layer1-dropout-`.

                    The dropout mode (training or not) is controlled
                    by the :attr:`mode` argument when calling the encoder.

                "dropout_rate" : float
                    The dropout rate, between 0 and 1. E.g.,
                    `"dropout_rate": 0.1` would drop out 10% of elements.

                "variational_dropout": bool
                    Whether the dropout mask is the same across all time steps.

            "output_layer_bw" : dict
                Hyperparameters of the backward output layer. Ignored if
                :attr:`output_layer_bw` is given in the constructor. Have the
                same structure and defaults with :attr:`"output_layer_fw"`.

            "output_layer_share_config" : bool
                Whether share hyperparameters of the backward output layer
                with the forward output layer. Note that the layer parameters
                are not shared.

                If `True` (default), :attr:`"output_layer_bw"` is ignored.

            "name" : str
                Name of the encoder
        """
        hparams = RNNEncoderBase.default_hparams()
        hparams.update({
            "rnn_cell_fw": layers.default_rnn_cell_hparams(),
            "rnn_cell_bw": layers.default_rnn_cell_hparams(),
            "rnn_cell_share_config": True,
            "output_layer_fw": _default_output_layer_hparams(),
            "output_layer_bw": _default_output_layer_hparams(),
            "output_layer_share_config": True,
            "name": "bidirectional_rnn_encoder"
        })
        return hparams

    def _build(self,
               inputs,
               sequence_length=None,
               initial_state_fw=None,
               initial_state_bw=None,
               time_major=False,
               mode=None,
               return_cell_output=False,
               **kwargs):
        """Encodes the inputs.

        Args:
            inputs: A 3D Tensor of shape `[batch_size, max_time, dim]`.
                The first two dimensions
                `batch_size` and `max_time` may be exchanged if
                `time_major=True` is specified.
            sequence_length (int list or 1D Tensor, optional): Sequence lengths
                of the batch inputs. Used to copy-through state and zero-out
                outputs when past a batch element's sequence length.
            initial_state (optional): Initial state of the RNN.
            time_major (bool): The shape format of the :attr:`inputs` and
                :attr:`outputs` Tensors. If `True`, these tensors are of shape
                `[max_time, batch_size, depth]`. If `False` (default),
                these tensors are of shape `[batch_size, max_time, depth]`.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, including
                `TRAIN`, `EVAL`, and `PREDICT`. Controls output layer dropout
                if the output layer is specified with :attr:`hparams`.
                If `None` (default), :func:`texar.context.global_mode()`
                is used.
            return_cell_output (bool): Whether to return the output of the RNN
                cell. This is the results prior to the output layer.
            **kwargs: Optional keyword arguments of
                :tf_main:`tf.nn.dynamic_rnn <nn/dynamic_rnn>`,
                such as `swap_memory`, `dtype`, `parallel_iterations`, etc.

        Returns:
            If :attr:`return_cell_output` is `False` (default), returns a
            pair :attr:`(outputs, final_state)` where

            - :attr:`outputs`: A tuple `(outputs_fw, outputs_bw)` containing \
              the forward and the backward RNN outputs, each of which is of \
              shape `[batch_size, max_time, output_dim]` (if \
              :attr:`time_major` == `False`) or \
              `[max_time, batch_size, output_dim]` (if \
              :attr:`time_major` == `True`). \

              If RNN cell output is a (nested) tuple of Tensors, then the \
              `outputs_fw` and `outputs_bw` will be a (nested) tuple having \
              the same structure as the cell output.

            - :attr:`final_state`: A tuple `(final_state_fw, final_state_bw)` \
              containing the final states of the forward and the backward \
              RNNs, each of which is a \
              Tensor of shape `[batch_size] + cell.state_size` or \
              a (nested) tuple of Tensors (if `cell.state_size` is a (nested) \
              tuple).

            If :attr:`return_cell_output` is `True`, returns a triple
            :attr:`(outputs, final_state, cell_outputs)` where

            - :attr:`cell_outputs`: A tuple \
              `(cell_outputs_fw, cell_outputs_bw)` containting the outputs \
              by the forward and backward RNN cells prior to the \
              output layers, having the same structure with :attr:`outputs` \
              except for the `output_dim`.
        """
        no_initial_state = initial_state_fw is None and initial_state_bw is None
        if ('dtype' not in kwargs) and no_initial_state:
            cell_outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self._cell_fw,
                cell_bw=self._cell_bw,
                inputs=inputs,
                sequence_length=sequence_length,
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
                time_major=time_major,
                dtype=tf.float32,
                **kwargs)
        else:
            cell_outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self._cell_fw,
                cell_bw=self._cell_bw,
                inputs=inputs,
                sequence_length=sequence_length,
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
                time_major=time_major,
                **kwargs)

        map_func_fw = functools.partial(
            _forward_output_layers,
            output_layer=self._output_layer_fw,
            time_major=time_major,
            hparams=self._hparams.output_layer_fw,
            mode=mode)
        outputs_fw = nest.map_structure(map_func_fw, cell_outputs[0])

        hparams_output_layer_bw = self._hparams.output_layer_bw
        if self._hparams.output_layer_share_config:
            hparams_output_layer_bw = self._hparams.output_layer_fw
        map_func_bw = functools.partial(
            _forward_output_layers,
            output_layer=self._output_layer_bw,
            time_major=time_major,
            hparams=hparams_output_layer_bw,
            mode=mode)
        outputs_bw = nest.map_structure(map_func_bw, cell_outputs[1])

        outputs = (outputs_fw, outputs_bw)

        if not self._built:
            self._add_internal_trainable_variables()
            # Add trainable variables of cells and output layers
            # which may be constructed externally.
            self._add_trainable_variable(
                layers.get_rnn_cell_trainable_variables(self._cell_fw))
            self._add_trainable_variable(
                layers.get_rnn_cell_trainable_variables(self._cell_bw))
            if self._output_layer_fw and \
                    not isinstance(self._output_layer_fw, (list, tuple)):
                self._add_trainable_variable(
                    self._output_layer_fw.trainable_variables)
            if self._output_layer_bw and \
                    not isinstance(self._output_layer_bw, (list, tuple)):
                self._add_trainable_variable(
                    self._output_layer_bw.trainable_variables)
            self._built = True

        if return_cell_output:
            return outputs, output_states, cell_outputs
        else:
            return outputs, output_states

    @staticmethod
    def concat_outputs(outputs):
        """Concats the outputs of the bidirectional encoder into a single
        tensor.
        """
        return tf.concat(outputs, 2)

    @property
    def cell_fw(self):
        """The forward RNN cell.
        """
        return self._cell_fw

    @property
    def cell_bw(self):
        """The backward RNN cell.
        """
        return self._cell_bw

    @property
    def state_size_fw(self):
        """The state size of the forward encoder cell.

        Same as :attr:`encoder.cell_fw.state_size`.
        """
        return self.cell_fw.state_size

    @property
    def state_size_bw(self):
        """The state size of the backward encoder cell.

        Same as :attr:`encoder.cell_bw.state_size`.
        """
        return self.cell_bw.state_size

    @property
    def output_layer_fw(self):
        """The output layer of the forward RNN.
        """
        return self._output_layer_fw

    @property
    def output_layer_bw(self):
        """The output layer of the backward RNN.
        """
        return self._output_layer_bw
