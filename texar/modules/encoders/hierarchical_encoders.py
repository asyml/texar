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

from collections import Sequence

#TODO(zhiting): this is incomplete
__all__ = [
    "HierarchicalRNNEncoder"
]

class HierarchicalRNNEncoder(EncoderBase):
    """Wrapper for RNN encoder with 2 levels.

    Useful for encoding structured long sequences, e.g. paragraphs, dialogs,
    etc.

    Args:
       encoder_major (optional): The context-level encoder receiving final 
                                 states from utterance-level encoder as its
                                 inputs. If it is not specified, an encoder 
                                 is created as specified in 
                                :attr:`hparams["encoder_major"]`.

       encoder_minor (optional): The utterance-level encoder. If it is not 
                                 specified, an encoder is created as specified 
                                 in :attr:`hparams["encoder_minor"]`.

       hparams (optional): the hyperparameters.

    See :class:`~texar.modules.encoders.rnn_encoders.RNNEncoderBase` for the
    arguments, and :meth:`default_hparams` for the default hyperparameters.
    """

    def __init__(self, encoder_major=None, encoder_minor=None,
                 hparams=None):
        EncoderBase.__init__(self, hparams)

        encoder_major_hparams = utils.get_instance_kwargs(
            None, self._hparams.encoder_major_hparams)
        encoder_minor_hparams = utils.get_instance_kwargs(
            None, self._hparams.encoder_minor_hparams)

        if isinstance(encoder_major, EncoderBase):
            self._encoder_major = encoder_major
        else:
            with tf.variable_scope(self.variable_scope.name):
                with tf.variable_scope('encoder_major'):
                    self._encoder_major = utils.check_or_get_instance(
                        self._hparams.encoder_major_type,
                        encoder_major_hparams,
                        ['texar.modules.encoders', 'texar.custom'])
        if isinstance(encoder_minor, EncoderBase):
            self._encoder_minor = encoder_minor
        elif self._hparams.config_share:
            with tf.variable_scope(self.variable_scope.name):
                with tf.variable_scope('encoder_minor'):
                    self._encoder_minor = utils.check_or_get_instance(
                        self._hparams.encoder_major_type,
                        encoder_major_hparams,
                        ['texar.modules.encoders', 'texar.custom'])
        else:
            with tf.variable_scope(self.variable_scope.name):
                with tf.variable_scope('encoder_minor'):
                    self._encoder_minor = utils.check_or_get_instance(
                        self._hparams.encoder_minor_type,
                        encoder_minor_hparams,
                        ['texar.modules.encoders', 'texar.custom'])

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        The dictionary has the following structure and default values.

        Returns:
            dict: Adictionary with following structure and values:
            .. code-block:: python
                {
                    "encoder_major_type": "UnidirectionalRNNEncoder",
                    "encoder_major_hparams": {},
                    "encoder_minor_type": "UnidirectionalRNNEncoder",
                    "encoder_minor_hparams": {},
                    "config_share": False,
                    "name": "hierarchical_encoder_wrapper"
                }

            Here:
        
            "encoder_major_type": 
                The class name of major encoder which can be found in 
                ~texar.modules.encoders or ~texar.custom.

            "encoder_major_hparams":
                The hparams for major encoder's construction.

            "config_share":
                :attr:`encoder_minor_type` and :attr:`encoder_minor_hparams`
                will be replaced by major's corresponding hparams if set to true.

            "name":
                Name of the encoder.
        """
        hparams = {
            "name": "hierarchical_encoder",
            "encoder_major_type": "UnidirectionalRNNEncoder",
            "encoder_major_hparams": {},
            "encoder_minor_type": "UnidirectionalRNNEncoder",
            "encoder_minor_hparams": {},
            "config_share": False,
            "@no_typecheck": [
                'encoder_major_hparams',
                'encoder_minor_hparams']
        }
        hparams.update(EncoderBase.default_hparams())
        return hparams

    def _build(self, inputs, order='btu',
               medium=None, **kwargs):
        """Encodes the inputs.

        Args:
            inputs: A 4D tensor of shape [B, T, U, dim], where
                        B: batch_size
                        T: the major seq len (context-level length)
                        U: the minor seq len (utterance-level length)

                    The order of first three dimensions can be changed
                    regarding to :attr:`time_major` of the two encoders.

            order (optional): a 3-char string with some order of 'b', 't', 'u',
                              specifying the order of inputs dimension.
                              Following four can be accepted:

                              'btu': time_major=False for both. (default)
                              'utb': time_major=True for both.
                              'tbu': time_major=True for major encoder only.
                              'ubt': time_major=True for minor encoder only.

            medium (optional): A callable function processes the final states of 
                               minor encoder to be the input for major encoder.
                               Extra meta like speaker token can be added using 
                               this function.
                               If not specified, a final state will be flatten 
                               into a vector while hidden part of LSTMTuple is 
                               skipped, see :meth:`depack_lstmtuple` for the scheme.

                               Use :attr:`states_minor_before_medium` and 
                               :attr:`states_minor_after_medium` to see its input
                               and output respectively.

            **kwargs: Optional keyword arguments of `tensorflow.nn.dynamic_rnn`,
                      such as `sequence_length`, `initial_state`, etc.

                      By default, arguments except `initial_state` and 
                      `sequence_length` will be sent to both major and minor 
                      encoders. To specify the encoder that arguments sent to, add 
                      '_minor'/'_major' as its suffix. 

                      `initial_state` and `sequence_length` will be sent to minor
                      encoder only if not specifing its encoder.

                      `initial_state` and `sequence_length` sent to minor encoder 
                      can be either 1-D tensor or 2-D tensor, with BxT units following
                      correct order.

        Returns:
            Outputs and final state of the major encoder.
        
        """

        def kwargs_split(kwargs):
            kwargs_minor, kwargs_major = {}, {}
            for k, v in kwargs.items():
                if len(k) < 5 or (k[-5:] not in ['major', 'minor']):
                    kwargs_minor[k] = v
                    if k not in ['sequence_length', 'initial_state']:
                        kwargs_major[k] = v
                    else:
                        kwargs_minor[k] = tf.reshape(v, [-1])
            for k, v in kwargs.items():
                if len(k) >= 6 and k[-6:] == ['_minor']:
                    kwargs_minor[k[:-6]] = v
                if len(k) >= 6 and k[-6:] == ['_major']:
                    kwargs_major[k[:-6]] = v

            return kwargs_minor, kwargs_major

        kwargs_minor, kwargs_major = kwargs_split(kwargs)

        shape = tf.shape(inputs)[:3]

        expand, shape = self._get_flatten_order(
            order, kwargs_major, kwargs_minor, tf.shape(inputs))

        inputs = tf.reshape(inputs, shape + [inputs.shape[3]])

        outputs_minor, states_minor = self._encoder_minor(inputs,
                                                          **kwargs_minor)

        self.states_minor_before_medium = states_minor

        if medium is None:
            states_minor = self.depack_lstmtuple(states_minor)
        else:
            states_minor = medium(states_minor)

        self.states_minor_after_medium = states_minor

        states_minor = tf.reshape(
            states_minor, tf.concat([expand, tf.shape(states_minor)[1:]], 0))

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

    @staticmethod
    def _get_flatten_order(order, kwargs_minor, kwargs_major, shape):
        def error_message(order):
            return ('Fail to match input order \'{}\'' \
                    'with given `time_major` params.').format(order)

        time_major_minor = kwargs_minor.get('time_major', None)
        time_major_major = kwargs_major.get('time_major', None)
        if order == 'btu':
            if not ((time_major_minor is None or not time_major_minor) and \
                    (time_major_major is None or not time_major_major)):
                raise ValueError(error_message(order))
            kwargs_minor.setdefault('time_major', False)
            kwargs_major.setdefault('time_major', False)
            expand = shape[0:2]
            shape = [shape[0] * shape[1], shape[2]]
        elif order == 'utb':
            if not ((time_major_minor is None or time_major_minor) and \
                    (time_major_major is None or time_major_major)):
                raise ValueError(error_message(order))
            kwargs_minor.setdefault('time_major', True)
            kwargs_major.setdefault('time_major', True)
            expand = shape[1:3]
            shape = [shape[0], shape[1] * shape[2]]
        elif order == 'tbu':
            if not ((time_major_minor is None or not time_major_minor) and \
                    (time_major_major is None or time_major_major)):
                raise ValueError(error_message(order))
            kwargs_minor.setdefault('time_major', False)
            kwargs_major.setdefault('time_major', True)
            expand = shape[0:2]
            shape = [shape[0] * shape[1], shape[2]]
        elif order == 'ubt':
            if not ((time_major_minor is None or time_major_minor) and \
                    (time_major_major is None or not time_major_major)):
                raise ValueError(error_message(order))
            kwargs_minor.setdefault('time_major', True)
            kwargs_major.setdefault('time_major', False)
            expand = shape[1:3]
            shape = [shape[0], shape[1] * shape[2]]

        return expand, shape

    @staticmethod
    def depack_lstmtuple(x):
        if isinstance(x, LSTMStateTuple):
            return x.h
        if isinstance(x, collections.Sequence):
            return tf.concat(
                [HierarchicalRNNEncoder.depack_lstmtuple(v) for v in x], -1)
        else:
            return x

    @property
    def encoder_major(self):
        return self._encoder_major
    
    @property
    def encoder_minor(self):
        return self._encoder_minor
