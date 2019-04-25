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
Base class for models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar import HParams

# pylint: disable=too-many-arguments

__all__ = [
    "ModelBase"
]


class ModelBase(object):
    """Base class inherited by all model classes.

    A model class implements interfaces that are compatible with
    :tf_main:`TF Estimator <estimator/Estimator>`. In particular,
    :meth:`_build` implements the
    :tf_main:`model_fn <estimator/Estimator#__init__>` interface; and
    :meth:`get_input_fn` is for the :attr:`input_fn` interface.

    .. document private functions
    .. automethod:: _build
    """

    def __init__(self, hparams=None):
        self._hparams = HParams(hparams, self.default_hparams(),
                                allow_new_hparam=True)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        hparams = {
            "name": "model"
        }
        return hparams

    def __call__(self, features, labels, params, mode, config=None):
        """Used for the :tf_main:`model_fn <estimator/Estimator#__init__>`
        argument when constructing
        :tf_main:`tf.estimator.Estimator <estimator/Estimator>`.
        """
        return self._build(features, labels, params, mode, config=config)

    def _build(self, features, labels, params, mode, config=None):
        """Used for the :tf_main:`model_fn <estimator/Estimator#__init__>`
        argument when constructing
        :tf_main:`tf.estimator.Estimator <estimator/Estimator>`.
        """
        raise NotImplementedError

    def get_input_fn(self, *args, **kwargs):
        """Returns the :attr:`input_fn` function that constructs the input
        data, used in :tf_main:`tf.estimator.Estimator <estimator/Estimator>`.
        """
        raise NotImplementedError

    @property
    def hparams(self):
        """A :class:`~texar.HParams` instance. The hyperparameters
        of the module.
        """
        return self._hparams

