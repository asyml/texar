# Copyright 2019 The Texar Authors. All Rights Reserved.
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
XLNet Regressor.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from texar.utils.mode import is_train_mode
from texar.core import layers
from texar.modules.regressors.regressor_base import RegressorBase
from texar.modules import XLNetEncoder
from texar.utils import utils
from texar.hyperparams import HParams

# pylint: disable=too-many-arguments, invalid-name, no-member,
# pylint: disable=too-many-branches, too-many-locals, too-many-statements

__all__ = [
    "XLNetRegressor"
]


class XLNetRegressor(RegressorBase):
    """Regressor based on XLNet modules.

    This is a combination of the :class:`~texar.modules.XLNetEncoder` with a
    classification layer. Both step-wise classification and sequence-level
    classification are supported, specified in :attr:`hparams`.

    Arguments are the same as in :class:`~texar.modules.XLNetEncoder`.

    Args:
        pretrained_model_name (optional): a str with the name
            of a pre-trained model to load. Currently only 'xlnet-large-cased'
            is supported. If `None`, will use the model name in :attr:`hparams`.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.

    .. document private functions
    .. automethod:: _build
    """

    def __init__(self,
                 pretrained_model_name=None,
                 cache_dir=None,
                 hparams=None):
        RegressorBase.__init__(self, hparams)

        with tf.variable_scope(self.variable_scope):
            tf.get_variable_scope().set_initializer(
                layers.get_initializer(self._hparams.initializer))
            # Creates the underlying encoder
            encoder_hparams = utils.dict_fetch(
                hparams, XLNetEncoder.default_hparams())
            if encoder_hparams is not None:
                encoder_hparams['name'] = "encoder"
            self._encoder = XLNetEncoder(
                pretrained_model_name=pretrained_model_name,
                cache_dir=cache_dir,
                hparams=encoder_hparams)
            if self._hparams.use_projection:
                self.projection = layers.get_layer(hparams={
                    "type": "Dense",
                    "kwargs": {
                        "units": self._encoder.output_size
                    }
                })

            # Creates an dropout layer
            drop_kwargs = {"rate": self._hparams.dropout}
            layer_hparams = {"type": "Dropout", "kwargs": drop_kwargs}
            self._dropout_layer = layers.get_layer(hparams=layer_hparams)

            logit_kwargs = self._hparams.logit_layer_kwargs
            if logit_kwargs is None:
                logit_kwargs = {}
            elif not isinstance(logit_kwargs, HParams):
                raise ValueError(
                    "hparams['logit_layer_kwargs'] must be a dict.")
            else:
                logit_kwargs = logit_kwargs.todict()
            logit_kwargs.update({"units": 1})
            if 'name' not in logit_kwargs:
                logit_kwargs['name'] = "logit_layer"

            layer_hparams = {"type": "Dense", "kwargs": logit_kwargs}
            self._logit_layer = layers.get_layer(hparams=layer_hparams)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                # (1) Same hyperparameters as in XLNetEncoder
                ...
                # (2) Additional hyperparameters
                "regr_strategy": "cls_time",
                "use_projection": True,
                "logit_layer_kwargs": None,
                "name": "xlnet_regressor",
            }

        Here:

        1. Same hyperparameters as in
           :class:`~texar.modules.XLNetEncoder`.
           See the :meth:`~texar.modules.XLNetEncoder.default_hparams`.
           An instance of XLNetEncoder is created for feature extraction.

        2. Additional hyperparameters:

            `"regr_strategy"`: str
                The regression strategy, one of:

                - **cls_time**: Sequence-level regression based on the
                  output of the first time step (which is the `CLS` token).
                  Each sequence has a prediction.
                - **all_time**: Sequence-level regression based on
                  the output of all time steps. Each sequence has a prediction.
                - **time_wise**: Step-wise regression, i.e., make
                  regression for each time step based on its output.

            `"logit_layer_kwargs"` : dict
                Keyword arguments for the logit Dense layer constructor,
                except for argument "units" which is set to "num_classes".
                Ignored if no extra logit layer is appended.

            `"use_projection"`: bool
                If `True`, an additional :torch_nn:`Linear` layer is added after
                the summary step.

            `"name"`: str
                Name of the regressor.
        """
        hparams = XLNetEncoder.default_hparams()
        hparams.update({
            "logit_layer_kwargs": None,
            "regr_strategy": "cls_time",
            "dropout": 0.1,
            "use_projection": True,
            "name": "xlnet_regressor"
        })
        return hparams

    def _build(self, token_ids, segment_ids=None, input_mask=None, mode=None):
        r"""Feeds the inputs through the network and makes regression.

        Args:
            token_ids: Shape `[batch_size, max_time]`.
            segment_ids: Shape `[batch_size, max_time]`.
            input_mask: Float tensor of shape `[batch_size, max_time]`. Note
                that positions with value 1 are masked out.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`,
                including `TRAIN`, `EVAL`, and `PREDICT`. Used to toggle
                dropout.
                If `None` (default), :func:`texar.global_mode` is used.

        Returns:
            Regression predictions.

            - If ``regr_strategy`` is ``cls_time`` or ``all_time``, predictions
              have shape `[batch_size]`.

            - If ``clas_strategy`` is ``time_wise``, predictions have shape
              `[batch_size, max_time]`.
        """
        is_training = is_train_mode(mode)
        output, _ = self._encoder(token_ids, segment_ids, input_mask=input_mask,
                                  mode=mode)

        strategy = self._hparams.regr_strategy
        if strategy == "time_wise":
            summary = output
        elif strategy == "cls_time":
            summary = output[:, -1]
        elif strategy == "all_time":
            length_diff = self._hparams.max_seq_len - tf.shape(token_ids)[1]
            summary_input = tf.pad(output,
                                   paddings=[[0, 0], [0, length_diff], [0, 0]])
            summary_input_dim = \
                self._encoder.output_size * self._hparams.max_seq_len
            summary = tf.reshape(summary_input, shape=[-1, summary_input_dim])
        else:
            raise ValueError("Unknown classification strategy: {}".
                             format(strategy))

        if self._hparams.use_projection:
            summary = tf.tanh(self.projection(summary))

        # summary: (batch_size, hidden_dim)
        summary = self._dropout_layer(summary, training=is_training)

        logits = tf.squeeze(self._logit_layer(summary), -1)

        if not self._built:
            self._add_internal_trainable_variables()
            if self._logit_layer:
                self._add_trainable_variable(
                    self._logit_layer.trainable_variables)
            self._built = True

        return logits
