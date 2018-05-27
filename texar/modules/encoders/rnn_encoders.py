#
"""
Various RNN encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.modules.encoders.encoder_base import EncoderBase
from texar.modules.networks.conv_networks import _to_list
from texar.core import layers
from texar.hyperparams import HParams

# pylint: disable=not-context-manager, too-many-arguments

__all__ = [
    "RNNEncoderBase",
    "UnidirectionalRNNEncoder",
    "BidirectionalRNNEncoder"
]

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
            output. If `None` (default), no output layer is applied.
        hparams (dict, optional): Encoder hyperparameters. If it is not
            specified, the default hyperparameter setting is used. See
            :attr:`default_hparams` for the sturcture and default values.
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
        self._output_layer = output_layer
        if output_layer is None:
            with tf.variable_scope(self.variable_scope):
                pass

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            .. code-block:: python

                {
                    "rnn_cell": default_rnn_cell_hparams(),
                    "output_layer": None,
                    "name": "unidirectional_rnn_encoder"
                }

            Here:

            "rnn_cell" : dict
                A dictionary of RNN cell hyperparameters. Ignored if
                :attr:`cell` is given when constructing the encoder.

                The default value is defined in
                :func:`~texar.core.layers.default_rnn_cell_hparams`.

            "output_layer" : int or list or None
                Dense Output layer(s) applied to the RNN cell output.

                - If `int`, a :tf_main:`Dense <layers/Dense>` layer of the \
                  layer size is added to the RNN cell output.
                - If `list` of `int`, a series of Dense layers of the layer \
                  sizes are stacked and added to the RNN cell output.
                - If `None`, no output layer is added.

                Ignored if :attr:`output_layer` is given in the constructor.

            "output_layer_dropout_rate" : float
                The dropout rate to apply to the output of each of the
                output layers except the final layer. (Do not include the RNN
                cell output). The dropout mode (training or not) is controlled
                by the :attr:`mode` argument when calling the encoder.

                This is used only when there are more than one output
                layers, i.e., :attr:`"output_layer"` is a list of length > 1.

                The default is `1.0`, which disables dropout.
                Ignored if :attr:`output_layer` is given in the constructor.

            "name" : str
                Name of the encoder
        """
        hparams = RNNEncoderBase.default_hparams()
        hparams.update({
            "rnn_cell": layers.default_rnn_cell_hparams(),
            "output_layer": {
                "num_layers": 0,
                "layer_size": 128,
                "activation": "identity",
                "final_layer_activation": None,
                "other_dense_kwargs": None,
                "dropout_layer_ids": [],
                "dropout_rate": 0.75,
            },
            "name": "unidirectional_rnn_encoder"
        })
        return hparams

    @staticmethod
    def _build_output_layer(hparams):
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

        layer_hparams = []
        for i in range(nlayers):


            activation = hparams.activation
            if i == nlayers - 1 and not hparams.final_layer_activation:
                activation = hparams.final_layer_activation

            kwargs_i = {"units": layer_size[i],
                        "activation": activation,
                        "name": "dense_%d" % (i+1)}
            kwargs_i.update(other_kwargs)

            layer_hparams.append({"type": "Dense", "kwargs": kwargs_i})



    def _build(self, inputs, sequence_length=None, initial_state=None,
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
            **kwargs: Optional keyword arguments of
                :tf_main:`tf.nn.dynamic_rnn <nn/dynamic_rnn>`,
                such as `time_major`, `dtype`, etc.

        Returns:
            Outputs and final state of the encoder.
        """
        #TODO(zhiting): add docs of 'Returns'
        if ('dtype' not in kwargs) and (initial_state is None):
            outputs, state = tf.nn.dynamic_rnn(
                cell=self._cell,
                inputs=inputs,
                sequence_length=sequence_length,
                initial_state=initial_state,
                dtype=tf.float32,
                **kwargs)
        else:
            outputs, state = tf.nn.dynamic_rnn(
                cell=self._cell,
                inputs=inputs,
                sequence_length=sequence_length,
                initial_state=initial_state,
                **kwargs)

        if not self._built:
            self._add_internal_trainable_variables()
            # Add trainable variables of `self._cell` which may be constructed
            # externally.
            self._add_trainable_variable(
                layers.get_rnn_cell_trainable_variables(self._cell))
            self._built = True

        return outputs, state

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


class BidirectionalRNNEncoder(RNNEncoderBase):
    """Bidirectional forward-backward RNN encoder.

    Args:
        cell: (RNNCell, optional) If it is not specified,
            a cell is created as specified in :attr:`hparams["rnn_cell"]`.
        cell_dropout_mode (optional): A Tensor taking value of
            :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, which
            toggles dropout in the RNN cell (e.g., activates dropout in the
            TRAIN mode). If `None`, :func:`~texar.context.global_mode` is used.
            Ignored if :attr:`cell` is given.
        hparams (dict, optional): Encoder hyperparameters. If it is not
            specified, the default hyperparameter setting is used. See
            :attr:`default_hparams` for the sturcture and default values.
    """

    def __init__(self,
                 cell_fw=None,
                 cell_bw=None,
                 cell_dropout_mode=None,
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
            elif self.hparams.rnn_cell_share_config:
                self._cell_bw = layers.get_rnn_cell(
                    self._hparams.rnn_cell_fw, cell_dropout_mode)
            else:
                self._cell_bw = layers.get_rnn_cell(
                    self._hparams.rnn_cell_bw, cell_dropout_mode)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            .. code-block:: python

                {
                    "rnn_cell_fw": default_rnn_cell_hparams(),
                    "rnn_cell_bw": default_rnn_cell_hparams(),
                    "rnn_cell_share_config": True
                    "name": "bidirectional_rnn_encoder"
                }

            Here:

            "rnn_cell_fw" : dict
                A dictionary of hyperparameters of the forward RNN cell.
                Ignored if :attr:`cell_fw` is given when constructing
                the encoder.

                The default value is defined in
                :meth:`~texar.core.layers.default_rnn_cell_hparams`.

            "rnn_cell_bw" : dict
                A dictionary of hyperparameters of the backward RNN cell.
                Ignored if :attr:`cell_bw` is given when constructing
                the encoder, or if :attr:`"rnn_cell_share_config"` is `True`.

                The default value is defined in
                :meth:`~texar.core.layers.default_rnn_cell_hparams`.

            "rnn_cell_share_config" : bool
                Whether share hyperparameters of the backward cell with the
                forward cell.

                If `True` (default), :attr:`"rnn_cell_bw"` is ignored.

            "name" : str
                Name of the encoder
        """
        hparams = RNNEncoderBase.default_hparams()
        hparams["rnn_cell_fw"] = layers.default_rnn_cell_hparams()
        hparams["rnn_cell_share_config"] = True
        hparams["rnn_cell_bw"] = layers.default_rnn_cell_hparams()
        hparams["name"] = "bidirectional_rnn_encoder"
        return hparams

    #TODO(zhiting): add docs of 'Returns'
    def _build(self, inputs, sequence_length=None,
               initial_state_fw=None, initial_state_bw=None, **kwargs):
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
            **kwargs: Optional keyword arguments of
                :tf_main:`tf.nn.dynamic_rnn <nn/dynamic_rnn>`,
                such as `time_major`, `dtype`, etc.

        Returns:
            Outputs and final state of the encoder.
        """
        no_initial_state = initial_state_fw is None and initial_state_bw is None
        if ('dtype' not in kwargs) and no_initial_state:
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self._cell_fw,
                cell_bw=self._cell_bw,
                inputs=inputs,
                sequence_length=sequence_length,
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
                dtype=tf.float32,
                **kwargs)
        else:
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self._cell_fw,
                cell_bw=self._cell_bw,
                inputs=inputs,
                sequence_length=sequence_length,
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
                **kwargs)

        if not self._built:
            self._add_internal_trainable_variables()
            # Add trainable variables of cells which may be constructed
            # externally.
            self._add_trainable_variable(
                layers.get_rnn_cell_trainable_variables(self._cell_fw))
            self._add_trainable_variable(
                layers.get_rnn_cell_trainable_variables(self._cell_bw))
            self._built = True

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
