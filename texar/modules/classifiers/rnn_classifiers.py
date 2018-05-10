#
"""
Various RNN classifiers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.modules.classifiers.classifier_base import ClassifierBase
from texar.modules.encoders.rnn_encoders import UnidirectionalRNNEncoder
from texar.core import layers
from texar.utils import utils

# pylint: disable=too-many-arguments

__all__ = [
]

#def RNNClassifierBase(ClassifierBase):
#    """Base class inherited by all RNN classifiers.
#    """
#
#    def __init__(self, hparams=None):
#        ClassifierBase.__init__(self, hparams)


class UnidirectionalRNNClassifier(ClassifierBase):
    """One directional RNN classifier.
    """

    def __init__(self,
                 cell=None,
                 cell_dropout_mode=None,
                 num_classes=None,
                 output_layer=None,
                 hparams=None):
        ClassifierBase.__init__(self, hparams)

        with tf.variable_scope(self.variable_scope):
            encoder_hparams = utils.fetch_subdict(
                hparams, UnidirectionalRNNEncoder.default_hparams())
            self._encoder = UnidirectionalRNNEncoder(
                cell=cell,
                cell_dropout_mode=cell_dropout_mode,
                hparams=encoder_hparams)

            # Creates an additional output layer if needed
            if num_classes is None:
                num_classes = self._hparams.num_classes
            self._num_classes = num_classes
            if output_layer is None:
                #TODO
                pass

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        hparams = UnidirectionalRNNEncoder.default_hparams()
        hparams.update({
            "num_classes": 2,
            "sequence_classification": False,
            "name": "unidirectional_rnn_classifier"
        })
        return hparams
