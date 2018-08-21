# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
    This is a combination of the
    :class:`~texar.modules.UnidirectionalRNNEncoder` with a classification
    layer. Both step-wise classification and sequence-level classification
    are supported, specified in :attr:`hparams`.

    Arguments are the same as in
    :class:`~texar.modules.UnidirectionalRNNEncoder`.

    Args:
        cell: (RNNCell, optional) If not specified,
            a cell is created as specified in :attr:`hparams["rnn_cell"]`.
        cell_dropout_mode (optional): A Tensor taking value of
            :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, which
            toggles dropout in the RNN cell (e.g., activates dropout in
            TRAIN mode). If `None`, :func:`~texar.global_mode` is used.
            Ignored if :attr:`cell` is given.
        output_layer (optional): An instance of
            :tf_main:`tf.layers.Layer <layers/Layer>`. Applies to the RNN cell
            output of each step. If `None` (default), the output layer is
            created as specified in :attr:`hparams["output_layer"]`.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    .. document private functions
    .. automethod:: _build
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

        .. code-block:: python

            {
                # (1) Same hyperparameters as in UnidirectionalRNNEncoder
                ...

                # (2) Additional hyperparameters
                "num_classes": 2,
                "logit_layer_kwargs": None,
                "clas_strategy": "final_time",
                "max_seq_length": None,
                "name": "unidirectional_rnn_classifier"
            }

        Here:

        1. Same hyperparameters as in
        :class:`~texar.modules.UnidirectionalRNNEncoder`.
        See the :meth:`~texar.modules.UnidirectionalRNNEncoder.default_hparams`.
        An instance of UnidirectionalRNNEncoder is created for feature
        extraction.

        2. Additional hyperparameters:

            "num_classes" : int
                Number of classes:

                - If **`> 0`**, an additional :tf_main:`Dense <layers/Dense>` \
                layer is appended to the encoder to compute the logits over \
                classes.
                - If **`<= 0`**, no dense layer is appended. The number of \
                classes is assumed to be the final dense layer size of the \
                encoder.

            "logit_layer_kwargs" : dict
                Keyword arguments for the logit Dense layer constructor,
                except for argument "units" which is set to "num_classes".
                Ignored if no extra logit layer is appended.

            "clas_strategy" : str
                The classification strategy, one of:

                - **"final_time"**: Sequence-leve classification based on \
                the output of the final time step. One sequence has one class.
                - **"all_time"**: Sequence-level classification based on \
                the output of all time steps. One sequence has one class.
                - **"time_wise"**: Step-wise classfication, i.e., make \
                classification for each time step based on its output.

            "max_seq_length" : int, optional
                Maximum possible length of input sequences. Required if
                "clas_strategy" is "all_time".

            "name" : str
                Name of the classifier.
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
        """Feeds the inputs through the network and makes classification.

        The arguments are the same as in
        :class:`~texar.modules.UnidirectionalRNNEncoder`.

        Args:
            inputs: A 3D Tensor of shape `[batch_size, max_time, dim]`.
                The first two dimensions
                `batch_size` and `max_time` may be exchanged if
                `time_major=True` is specified.
            sequence_length (optional): A 1D int tensor of shape `[batch_size]`.
                Sequence lengths
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
                If `None` (default), :func:`texar.global_mode()`
                is used.
            return_cell_output (bool): Whether to return the output of the RNN
                cell. This is the results prior to the output layer.
            **kwargs: Optional keyword arguments of
                :tf_main:`tf.nn.dynamic_rnn <nn/dynamic_rnn>`,
                such as `swap_memory`, `dtype`, `parallel_iterations`, etc.

        Returns:
            A tuple `(logits, pred)`, containing the logits over classes and
            the predictions, respectively.

            - If "clas_strategy"=="final_time" or "all_time"

                - If "num_classes"==1, `logits` and `pred` are of both \
                shape `[batch_size]`
                - If "num_classes">1, `logits` is of shape \
                `[batch_size, num_classes]` and `pred` is of shape \
                `[batch_size]`.

            - If "clas_strategy"=="time_wise",

                - If "num_classes"==1, `logits` and `pred` are of both \
                shape `[batch_size, max_time]`
                - If "num_classes">1, `logits` is of shape \
                `[batch_size, max_time, num_classes]` and `pred` is of shape \
                `[batch_size, max_time]`.
                - If `time_major` is `True`, the batch and time dimensions are\
                exchanged.
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
        num_classes = self._hparams.num_classes
        is_binary = num_classes == 1
        is_binary = is_binary or (num_classes <= 0 and logits.shape[-1] == 1)

        if stra == 'time_wise':
            if is_binary:
                pred = tf.squeeze(tf.greater(logits, 0), -1)
                logits = tf.squeeze(logits, -1)
            else:
                pred = tf.argmax(logits, axis=-1)
        else:
            if is_binary:
                pred = tf.greater(logits, 0)
                logits = tf.reshape(logits, [-1])
            else:
                pred = tf.argmax(logits, axis=-1)
            pred = tf.reshape(pred, [-1])
        pred = tf.to_int64(pred)

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
