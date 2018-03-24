#
"""
Various encoders that encode data with hierarchical structure.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple

from texar.modules.encoders import UnidirectionalRNNEncoder
from texar.modules.encoders.encoder_base import EncoderBase
from texar.utils import utils

#TODO(zhiting): this is incomplete
__all__ = [
    "HierarchicalEncoder"
]

from IPython import embed

class HierarchicalEncoder(EncoderBase):
    """One directional forward RNN encoder with 2 levels.

    Useful for encoding structured long sequences, e.g. paragraphs, dialogs,
    etc.

    Expect 3D tensor input [B, T, U, D] where
    B: batch size
    T: the seq len along the major (context-level) encoder (MAJOR)
    U: the seq len along the minor (utterance   -level) encoder (MINOR)
    D: the dimension of each element fed into the minor encoder

    The pipeline is to simply feed the input with shape [B X T, U, D] into
    MINOR, and then regard the final_state with shape [B, T, D'] as
    the input of MAJOR, where D' is the dims of MINOR's hidden state.

    If time_major is used, then input becomes [U, T, B, D] (reverse the first
    three dims).

    the minor encoder supports various types: RNN, bi-RNN, CNN, CBOW etc.

    Args:
       encoder_major (optional): if not given it then use the setting specified
                                 in hparams.
       encoder_minor (optional): ditto.
       hparams (optional): the hyperparameters.

    See :class:`~texar.modules.encoders.rnn_encoders.RNNEncoderBase` for the
    arguments, and :meth:`default_hparams` for the default hyperparameters.
    """

    def __init__(self,
                 encoder_major=None,
                 encoder_minor=None,
                 hparams=None):
        EncoderBase.__init__(self, hparams)

        if isinstance(encoder_major, EncoderBase):
            self._encoder_major = encoder_major
        else:
            cls = utils.get_class(self._hparams.encoder_major.class_name,
                                  ['texar.modules.encoders', 'texar.custom'])
            with tf.variable_scope(self.variable_scope.name):
                with tf.variable_scope('major'):
                    self._encoder_major = cls(
                        hparams=self._hparams.encoder_major.hparams)

        if isinstance(encoder_minor, EncoderBase):
            self._encoder_minor = encoder_minor
        elif self._hparams.encoder_minor.share_config:
            cls = utils.get_class(self._hparams.encoder_major.class_name,
                                  ['texar.modules.encoders', 'texar.custom'])
            with tf.variable_scope(self.variable_scope.name):
                with tf.variable_scope('minor'):
                    self._encoder_minor = cls(
                        hparams=self._hparams.encoder_major.hparams)
        else:
            cls = utils.get_class(self._hparams.encoder_minor.class_name,
                                  ['texar.modules.encoders', 'texar.custom'])
            with tf.variable_scope(self.variable_scope.name):
                with tf.variable_scope('minor'):
                    self._encoder_minor = cls(
                        hparams=self._hparams.encoder_minor.hparams)

    #TODO(zhiting): docs for hparams `minor_type` and `minor_cell`.
    #TODO(xingjiang):
    #   Unfortunately due to the check rule of hyperparameters we cannot
    #   specify the class_name of encoder, should be improved.
    #   See the docs for more details.
    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        The dictionary has the following structure and default values.

        (TODO)

        Returns:
            dict: Adictionary with following structure and values:
            .. code-block:: python

                {
                    "name": "hierarchical_encoder_wrapper"
                }
        """
        hparams = {
            "encoder_major": {
                "class_name": "UnidirectionalRNNEncoder",
                "hparams": UnidirectionalRNNEncoder.default_hparams(),
            },
            "encoder_minor": {
                "class_name": "UnidirectionalRNNEncoder",
                "hparams": UnidirectionalRNNEncoder.default_hparams(),
                "share_config": False,
            }
        }
        hparams.update(EncoderBase.default_hparams())
        hparams["name"] = "hierarchical_forward_rnn_encoder"
        return hparams

    def _build(self, inputs, **kwargs):
        """Encodes the inputs.

        Args:
            inputs: 4D tensor input [B, T, U, D] if time_major=false, otherwise
                    4D tensor input [U, T, B, D].
            **kwargs: Optional keyword arguments of `tensorflow.nn.dynamic_rnn`,
                such as `sequence_length`, `initial_state`, etc.

            Notice that if you want to specific some kwargs for either of
            MINOR/MAJOR encoder, add '_minor'/'_major' to the end of its key,
            otherwise it will be shared to both of them, except `initial_state`
            and `sequence_length`, which are only sent to MINOR encoder.

            `sequence_length` and `initial_state` for the MINOR encoder must be
            flatten manually, ensure that it follows the order [B, T] when
            time_major=false or [T, B] otherwise.

            Currently, `time_major` should be the same in both two encoders.

        Returns:
            Outputs and final state of the MAJOR encoder.
        """

        def kwargs_split(kwargs):
            kwargs_minor, kwargs_major = {}, {}
            for k, v in kwargs.items():
                if len(k) < 5 or (k[-5:] not in ['major', 'minor']):
                    kwargs_minor[k] = v
                    if k not in ['sequence_length', 'initial_state']:
                        kwargs_major[k] = v
            for k, v in kwargs.items():
                if len(k) >= 6 and k[-6:] == ['_minor']:
                    kwargs_minor[k[:-6]] = v
                if len(k) >= 6 and k[-6:] == ['_major']:
                    kwargs_major[k[:-6]] = v

            return kwargs_minor, kwargs_major

        kwargs_minor, kwargs_major = kwargs_split(kwargs)

        time_major = kwargs_minor.get('time_major', False)
        kwargs_major['time_major'] = time_major

        if time_major == True:
            batch_size = inputs.shape[2].value
            inputs = tf.reshape(inputs,
                                (inputs.shape[0], -1, inputs.shape[-1]))
        else:
            batch_size = inputs.shape[0].value
            inputs = tf.reshape(inputs,
                                (-1, inputs.shape[-2], inputs.shape[-1]))

        outputs_minor, states_minor = self._encoder_minor(inputs,
                                                          **kwargs_minor)
        if isinstance(states_minor, LSTMStateTuple):
            states_minor = states_minor.h

        if time_major == True:
            states_minor = tf.reshape(states_minor,
                                      (-1, batch_size, states_minor.shape[-1]))
        else:
            states_minor = tf.reshape(states_minor,
                                      (batch_size, -1, states_minor.shape[-1]))

        outputs_major, states_major = self._encoder_major(states_minor,
                                                          **kwargs_major)

        # Add trainable variables of `self._cell` which may be constructed
        # externally

        if self._built == False:
            self._add_trainable_variable(
                self._encoder_minor.trainable_variables)
            self._add_trainable_variable(
                self._encoder_major.trainable_variables)
            self._built = True

        return outputs_major, states_major

if __name__ == '__main__':
    "test script"
    a = HierarchicalEncoder()
    pass
