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
BERT classifiers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from texar.core import layers
from texar.modules.classifiers.classifier_base import ClassifierBase
from texar.modules import BertEncoder
from texar.utils import utils
from texar.hyperparams import HParams

# pylint: disable=too-many-arguments, invalid-name, no-member,
# pylint: disable=too-many-branches, too-many-locals, too-many-statements

__all__ = [
    "BertClassifier"
]


class BertClassifier(ClassifierBase):
    """Classifier based on bert modules.

    This is a combination of the
    :class:`~texar.modules.BertEncoder` with a classification
    layer. Both step-wise classification and sequence-level classification
    are supported, specified in :attr:`hparams`.

    Arguments are the same as in
    :class:`~texar.modules.BertEncoder`.

    Args:
        pretrained_model_name (optional): a str with the name
            of a pre-trained model to load selected in the list of:
            `bert-base-uncased`, `bert-large-uncased`, `bert-base-cased`,
            `bert-large-cased`, `bert-base-multilingual-uncased`,
            `bert-base-multilingual-cased`, `bert-base-chinese`.
            If `None`, will use the model name in :attr:`hparams`.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture
            and default values.

    .. document private functions
    .. automethod:: _build
    """

    def __init__(self,
                 pretrained_model_name=None,
                 cache_dir=None,
                 hparams=None):

        ClassifierBase.__init__(self, hparams)

        with tf.variable_scope(self.variable_scope):
            # Creates the underlying encoder
            encoder_hparams = utils.dict_fetch(
                hparams, BertEncoder.default_hparams())
            if encoder_hparams is not None:
                encoder_hparams['name'] = None
            self._encoder = BertEncoder(
                pretrained_model_name=pretrained_model_name,
                cache_dir=cache_dir,
                hparams=encoder_hparams)

            # Creates an dropout layer
            drop_kwargs = {"rate": self._hparams.dropout}
            layer_hparams = {"type": "Dropout", "kwargs": drop_kwargs}
            self._dropout_layer = layers.get_layer(hparams=layer_hparams)

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
                # (1) Same hyperparameters as in BertEncoder
                ...
                # (2) Additional hyperparameters
                "num_classes": 2,
                "logit_layer_kwargs": None,
                "clas_strategy": "cls_time",
                "max_seq_length": None,
                "dropout": 0.1,
                "name": "bert_classifier"
            }

        Here:

        1. Same hyperparameters as in
        :class:`~texar.modules.BertEncoder`.
        See the :meth:`~texar.modules.BertEncoder.default_hparams`.
        An instance of BertEncoder is created for feature extraction.

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
                - **"cls_time"**: Sequence-level classification based on the \
                output of the first time step (which is the "CLS" token). \
                Each sequence has a class.
                - **"all_time"**: Sequence-level classification based on \
                the output of all time steps. Each sequence has a class.
                - **"time_wise"**: Step-wise classfication, i.e., make \
                classification for each time step based on its output.

            "max_seq_length" : int, optional
                Maximum possible length of input sequences. Required if
                "clas_strategy" is "all_time".

            "dropout" : float
                The dropout rate of the bert encoder output.

            "name" : str
                Name of the classifier.
        """

        hparams = BertEncoder.default_hparams()
        hparams.update({
            "num_classes": 2,
            "logit_layer_kwargs": None,
            "clas_strategy": "cls_time",
            "max_seq_length": None,
            "dropout": 0.1,
            "name": "bert_classifier"
        })
        return hparams

    def _build(self,
               inputs,
               sequence_length=None,
               segment_ids=None,
               mode=None,
               **kwargs):
        """Feeds the inputs through the network and makes classification.

        The arguments are the same as in
        :class:`~texar.modules.BertEncoder`.

        Args:
            inputs: A 2D Tensor of shape `[batch_size, max_time]`,
                containing the token ids of tokens in input sequences.
            sequence_length (optional): A 1D Tensor of shape `[batch_size]`.
                Input tokens beyond respective sequence lengths are masked
                out automatically.
            segment_ids (optional): A 2D Tensor of shape
                `[batch_size, max_time]`, containing the segment ids
                of tokens in input sequences. If `None` (default), a tensor
                with all elements set to zero is used.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`,
                including `TRAIN`, `EVAL`, and `PREDICT`. Used to toggle
                dropout.
                If `None` (default), :func:`texar.global_mode` is used.
            **kwargs: Keyword arguments.

        Returns:
            A tuple `(logits, pred)`, containing the logits over classes and
            the predictions, respectively.

            - If "clas_strategy"=="cls_time" or "all_time"

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
        """

        enc_outputs, pooled_output = self._encoder(inputs, sequence_length,
                                                   segment_ids, mode)

        # Compute logits
        stra = self._hparams.clas_strategy
        if stra == 'time_wise':
            logits = enc_outputs
        elif stra == 'cls_time':
            logits = pooled_output
        elif stra == 'all_time':
            # Pad `enc_outputs` to have max_seq_length before flatten
            length_diff = self._hparams.max_seq_length - tf.shape(inputs)[1]
            length_diff = tf.reshape(length_diff, [1, 1])
            # Set `paddings = [[0, 0], [0, length_dif], [0, 0]]`
            paddings = tf.pad(length_diff, paddings=[[1, 1], [1, 0]])
            logit_input = tf.pad(enc_outputs, paddings=paddings)
            logit_input_dim = self._hparams.hidden_size * \
                              self._hparams.max_seq_length
            logits = tf.reshape(logit_input, [-1, logit_input_dim])
        else:
            raise ValueError('Unknown classification strategy: {}'.format(stra))

        if self._logit_layer is not None:
            logits = self._dropout_layer(logits, training=mode)
            logits = self._logit_layer(logits)

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
            if self._logit_layer:
                self._add_trainable_variable(
                    self._logit_layer.trainable_variables)
            self._built = True

        return logits, pred
