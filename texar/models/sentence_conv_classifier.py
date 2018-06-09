#
"""
Convolutional classifier for sentences.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.models import ModelBase
from texar.modules.classifiers import Conv1DClassifier
from texar.core import optimization as opt

class SentenceConvClassifier(ModelBase):
    """TODO
    """

    def __init__(self, hparams=None):
        ModelBase.__init__(self, hparams=hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        hparams = Conv1DClassifier.default_hparams()
        hparams.update({
            "optimization": opt.default_optimization_hparams(),
            "name": "sentence_conv_classifier"
        })
        return hparams
