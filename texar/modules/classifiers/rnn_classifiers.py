#
"""
Various RNN classifiers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.contrib.framework import nest

from texar.modules.classifiers.classifier_base import ClassifierBase
from texar.modules.encoders.rnn_encoders import \
        UnidirectionalRNNEncoder, _forward_single_output_layer
from texar.core import layers
from texar.utils import utils, shapes
from texar.hyperparams import HParams

# pylint: disable=too-many-arguments, invalid-name, no-member,
# pylint: disable=too-many-branches, too-many-locals, too-many-statements

__all__ = [
    "UnidirectionalRNNClassifier"
]

#def RNNClassifierBase(ClassifierBase):
#    """Base class inherited by all RNN classifiers.
#    """
#
#    def __init__(self, hparams=None):
#        ClassifierBase.__init__(self, hparams)


class UnidirectionalRNNClassifier(ClassifierBase):
    """One directional RNN classifier.

    Arguments are the same as in
    :class:`~texar.modules.UnidirectionalRNNEncoder`.

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
        ClassifierBase.__init__(self, hparams)

        with tf.variable_scope(self.variable_scope):
            # Creates the underlying encoder
            encoder_hparams = utils.dict_fetch(
                hparams, UnidirectionalRNNEncoder.default_hparams())
            if encoder_hparams is not None:
                encoder_hparams['name'] = None
            self._encoder = UnidirectionalRNNEncoder(
                cell=cell,
                cell_dropout_mode=cell_dropout_mode,
                output_layer=output_layer,
                hparams=encoder_hparams)

            # Creates an additional classification layer if needed
            self._num_classes = self._hparams.num_classes
            if self._num_classes <= 0:
                self._logit_layer = None
            else:
                logit_kwargs = self._hparams.logit_layer_kwargs
                if logit_kwargs is None:
                    logit_kwargs = {}
                elif not isinstance(logit_kwargs, HParams):
                    raise ValueError(
                        "hparams['logit_layer_kwargs'] must be a dict.")
                else:
                    logit_kwargs = logit_kwargs.todict()
                logit_kwargs.update({"units": self._num_classes})
                if 'name' not in logit_kwargs:
                    logit_kwargs['name'] = "logit_layer"

                layer_hparams = {"type": "Dense", "kwargs": logit_kwargs}
                self._logit_layer = layers.get_layer(hparams=layer_hparams)


    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        TODO
        final_time, all_time, time_wise,
        """
        hparams = UnidirectionalRNNEncoder.default_hparams()
        hparams.update({
            "num_classes": 2,
            "logit_layer_kwargs": None,
            "clas_strategy": "final_time",
            "max_seq_length": None,
            "name": "unidirectional_rnn_classifier"
        })
        return hparams

    def _build(self,
               inputs,
               sequence_length=None,
               initial_state=None,
               time_major=False,
               mode=None,
               **kwargs):
        """
        """
        enc_outputs, _, enc_output_size = self._encoder(
            inputs=inputs,
            sequence_length=sequence_length,
            initial_state=initial_state,
            time_major=time_major,
            mode=mode,
            return_output_size=True,
            **kwargs)

        # Flatten enc_outputs
        enc_outputs_flat = nest.flatten(enc_outputs)
        enc_output_size_flat = nest.flatten(enc_output_size)
        enc_output_dims_flat = [np.prod(xs) for xs in enc_output_size_flat]
        enc_outputs_flat = [shapes.flatten(x, 2, xs) for x, xs
                            in zip(enc_outputs_flat, enc_output_dims_flat)]
        if len(enc_outputs_flat) == 1:
            enc_outputs_flat = enc_outputs_flat[0]
        else:
            enc_outputs_flat = tf.concat(enc_outputs_flat, axis=2)

        # Compute logits
        stra = self._hparams.clas_strategy
        if stra == 'time_wise':
            logits = enc_outputs_flat
        elif stra == 'final_time':
            if time_major:
                logits = enc_outputs_flat[-1, :, :]
            else:
                logits = enc_outputs_flat[:, -1, :]
        elif stra == 'all_time':
            if self._logit_layer is None:
                raise ValueError(
                    'logit layer must not be `None` if '
                    'clas_strategy="all_time". Specify the logit layer by '
                    'either passing the layer in the constructor or '
                    'specifying the hparams.')
            if self._hparams.max_seq_length is None:
                raise ValueError(
                    'hparams.max_seq_length must not be `None` if '
                    'clas_strategy="all_time"')
        else:
            raise ValueError('Unknown classification strategy: {}'.format(stra))

        if self._logit_layer is not None:
            logit_input_dim = np.sum(enc_output_dims_flat)
            if stra == 'time_wise':
                logits, _ = _forward_single_output_layer(
                    logits, logit_input_dim, self._logit_layer)
            elif stra == 'final_time':
                logits = self._logit_layer(logits)
            elif stra == 'all_time':
                # Pad `enc_outputs_flat` to have max_seq_length before flatten
                length_diff = self._hparams.max_seq_length - tf.shape(inputs)[1]
                length_diff = tf.reshape(length_diff, [1, 1])
                # Set `paddings = [[0, 0], [0, length_dif], [0, 0]]`
                paddings = tf.pad(length_diff, paddings=[[1, 1], [1, 0]])
                logit_input = tf.pad(enc_outputs_flat, paddings=paddings)

                logit_input_dim *= self._hparams.max_seq_length
                logit_input = tf.reshape(logit_input, [-1, logit_input_dim])

                logits = self._logit_layer(logit_input)

        # Compute predications
        if stra == 'time_wise':
            pred = tf.argmax(logits, axis=-1)
        else:
            pred = tf.argmax(logits, axis=1)

        if not self._built:
            self._add_internal_trainable_variables()
            # Add trainable variables of `self._logit_layer`
            # which may be constructed externally.
            if self._logit_layer:
                self._add_trainable_variable(
                    self._logit_layer.trainable_variables)
            self._built = True

        return logits, pred

    @property
    def num_classes(self):
        """The number of classes, specified in :attr:`hparams`.
        """
        return self._hparams.num_classes
