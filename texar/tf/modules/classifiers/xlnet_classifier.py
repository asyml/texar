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
XLNet classifiers.
"""

import tensorflow as tf

from texar.tf.utils.mode import is_train_mode
from texar.tf.core.layers import get_layer, get_initializer
from texar.tf.modules.classifiers.classifier_base import ClassifierBase
from texar.tf.modules.encoders.xlnet_encoder import XLNetEncoder
from texar.tf.hyperparams import HParams
from texar.tf.modules.pretrained.xlnet import PretrainedXLNetMixin
from texar.tf.utils.utils import dict_fetch

# pylint: disable=too-many-arguments, invalid-name, no-member,
# pylint: disable=too-many-branches, too-many-locals, too-many-statements

__all__ = [
    "XLNetClassifier"
]


class XLNetClassifier(ClassifierBase, PretrainedXLNetMixin):
    """Classifier based on XLNet modules. Please see
    :class:`~texar.tf.modules.PretrainedXLNetMixin` for a brief description
    of XLNet.

    This is a combination of the :class:`~texar.tf.modules.XLNetEncoder` with a
    classification layer. Both step-wise classification and sequence-level
    classification are supported, specified in :attr:`hparams`.

    Arguments are the same as in :class:`~texar.tf.modules.XLNetEncoder`.

    Args:
        pretrained_model_name (optional): a `str`, the name
            of pre-trained model (e.g., ``xlnet-based-cased``). Please refer to
            :class:`~texar.tf.modules.PretrainedXLNetMixin` for
            all supported models.
            If `None`, the model name in :attr:`hparams` is used.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory (``texar_data`` folder under user's home
            directory) will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.

    .. document private functions
    .. automethod:: _build
    """

    def __init__(self,
                 pretrained_model_name=None,
                 cache_dir=None,
                 hparams=None):
        super(XLNetClassifier, self).__init__(hparams=hparams)

        with tf.variable_scope(self.variable_scope):
            tf.get_variable_scope().set_initializer(
                get_initializer(self._hparams.initializer))
            # Creates the underlying encoder
            encoder_hparams = dict_fetch(
                hparams, XLNetEncoder.default_hparams())
            if encoder_hparams is not None:
                encoder_hparams['name'] = "encoder"
            self._encoder = XLNetEncoder(
                pretrained_model_name=pretrained_model_name,
                cache_dir=cache_dir,
                hparams=encoder_hparams)
            if self._hparams.use_projection:
                self.projection = get_layer(hparams={
                    "type": "Dense",
                    "kwargs": {
                        "units": self._encoder.output_size
                    }
                })

            # Creates an dropout layer
            drop_kwargs = {"rate": self._hparams.dropout}
            layer_hparams = {"type": "Dropout", "kwargs": drop_kwargs}
            self._dropout_layer = get_layer(hparams=layer_hparams)

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
                self._logit_layer = get_layer(hparams=layer_hparams)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                # (1) Same hyperparameters as in XLNetEncoder
                ...
                # (2) Additional hyperparameters
                "clas_strategy": "cls_time",
                "use_projection": True,
                "num_classes": 2,
                "logit_layer_kwargs": None,
                "name": "xlnet_classifier",
            }

        Here:

        1. Same hyperparameters as in
            :class:`~texar.tf.modules.XLNetEncoder`.
            See the :meth:`~texar.tf.modules.XLNetEncoder.default_hparams`.
            An instance of XLNetEncoder is created for feature extraction.

        2. Additional hyperparameters:

            `"clas_strategy"`: str
                The classification strategy, one of:

                - **cls_time**: Sequence-level classification based on the
                  output of the last time step (which is the `CLS` token).
                  Each sequence has a class.
                - **all_time**: Sequence-level classification based on
                  the output of all time steps. Each sequence has a class.
                - **time_wise**: Step-wise classification, i.e., make
                  classification for each time step based on its output.

            `"use_projection"`: bool
                If `True`, an additional `Dense` layer is added after the
                summary step.

            `"num_classes"`: int
                Number of classes:

                - If **> 0**, an additional dense layer is appended to the
                  encoder to compute the logits over classes.
                - If **<= 0**, no dense layer is appended. The number of
                  classes is assumed to be the final dense layer size of the
                  encoder.

            `"logit_layer_kwargs"` : dict
                Keyword arguments for the logit Dense layer constructor,
                except for argument "units" which is set to "num_classes".
                Ignored if no extra logit layer is appended.

            `"name"`: str
                Name of the classifier.
        """
        hparams = XLNetEncoder.default_hparams()
        hparams.update({
            "num_classes": 2,
            "logit_layer_kwargs": None,
            "clas_strategy": "cls_time",
            "dropout": 0.1,
            "use_projection": True,
            "name": "xlnet_classifier"
        })
        return hparams

    def param_groups(self, lr=None, lr_layer_scale=1.0,
                     decay_base_params=False):
        r"""Create parameter groups for optimizers. When
        :attr:`lr_layer_decay_rate` is not 1.0, parameters from each layer form
        separate groups with different base learning rates.

        This method should be called before applying gradients to the variables
        through the optimizer. Particularly, after calling the optimizer's
        `compute_gradients` method, the user can call this method to get
        variable-specific learning rates for the network. The gradients for each
        variables can then be scaled accordingly. These scaled gradients are
        finally applied by calling optimizer's `apply_gradients` method.

        Args:
            lr (float): The learning rate. Can be omitted if
                :attr:`lr_layer_decay_rate` is 1.0.
            lr_layer_scale (float): Per-layer LR scaling rate. The `i`-th layer
                will be scaled by `lr_layer_scale ^ (num_layers - i - 1)`.
            decay_base_params (bool): If `True`, treat non-layer parameters
                (e.g. embeddings) as if they're in layer 0. If `False`, these
                parameters are not scaled.

        Returns: A dict mapping tensorflow variables to their learning rates.
        """
        vars_to_learning_rates = {}
        if lr_layer_scale != 1.0:
            if lr is None:
                raise ValueError(
                    "lr must be specified when lr_layer_decay_rate is not 1.0")

            scope = self.variable_scope.name
            projection_vars = tf.trainable_variables(scope=scope + "/dense")
            logits_vars = tf.trainable_variables(
                scope=self.variable_scope.name + "/logit_layer")
            finetune_vars = projection_vars + logits_vars
            for var in finetune_vars:
                vars_to_learning_rates[var] = lr

            vars_to_learning_rates.update(
                self._encoder.param_groups(lr=lr,
                                           lr_layer_scale=lr_layer_scale,
                                           decay_base_params=decay_base_params))
        else:
            for variable in self.trainable_variables:
                vars_to_learning_rates[variable] = lr

        return vars_to_learning_rates

    def _build(self, token_ids, segment_ids=None, input_mask=None, mode=None):
        r"""Feeds the inputs through the network and makes classification.

        Args:
            token_ids: Shape `[batch_size, max_time]`.
            segment_ids: Shape `[batch_size, max_time]`.
            input_mask: Float tensor of shape `[batch_size, max_time]`. Note
                that positions with value 1 are masked out.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`,
                including `TRAIN`, `EVAL`, and `PREDICT`. Used to toggle
                dropout.
                If `None` (default), :func:`texar.tf.global_mode` is used.

        Returns:
            A tuple `(logits, preds)`, containing the logits over classes and
            the predictions, respectively.

            - If ``clas_strategy`` is ``cls_time`` or ``all_time``:

                - If ``num_classes`` == 1, ``logits`` and ``pred`` are both of
                  shape ``[batch_size]``.
                - If ``num_classes`` > 1, ``logits`` is of shape
                  ``[batch_size, num_classes]`` and ``pred`` is of shape
                  ``[batch_size]``.

            - If ``clas_strategy`` is ``time_wise``:

                - ``num_classes`` == 1, ``logits`` and ``pred`` are both of
                  shape ``[batch_size, max_time]``.
                - If ``num_classes`` > 1, ``logits`` is of shape
                  ``[batch_size, max_time, num_classes]`` and ``pred`` is of
                  shape ``[batch_size, max_time]``.
        """
        is_training = is_train_mode(mode)
        output, _ = self._encoder(token_ids, segment_ids, input_mask=input_mask,
                                  mode=mode)
        strategy = self._hparams.clas_strategy
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
            raise ValueError("Unknown classification strategy: {}"
                             .format(strategy))

        if self._hparams.use_projection:
            summary = tf.tanh(self.projection(summary))
        # summary: (batch_size, hidden_dim)
        summary = self._dropout_layer(summary, training=is_training)

        logits = (self._logit_layer(summary) if self._logit_layer is not None
                  else summary)

        # Compute predictions
        num_classes = self._hparams.num_classes
        is_binary = num_classes == 1 or (num_classes <= 0
                                         and logits.shape[-1] == 1)

        if strategy == "time_wise":
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
            if self._logit_layer:
                self._add_trainable_variable(
                    self._logit_layer.trainable_variables)
            self._built = True

        return logits, pred
