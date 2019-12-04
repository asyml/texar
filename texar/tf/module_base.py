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
Base class for modules.
"""
from abc import ABC, abstractmethod

import tensorflow as tf

from texar.tf.hyperparams import HParams

__all__ = [
    "ModuleBase",
]


class ModuleBase(tf.keras.layers.Layer, ABC):
    r"""Base class inherited by modules that create Variables and are
    configurable through hyperparameters.

    A Texar module inheriting :class:`~texar.tf.ModuleBase` is
    **configurable through hyperparameters**. That is, each module defines
    allowed hyperparameters and default values. Hyperparameters not
    specified by users will take default values.

    Args:
        hparams (dict, optional): Hyperparameters of the module. See
            :meth:`default_hparams` for the structure and default values.
    """

    def __init__(self, hparams=None):
        super().__init__()
        if not hasattr(self, '_hparams'):
            self._hparams = HParams(hparams, self.default_hparams())
        else:
            # Probably already parsed by subclasses. We rely on subclass
            # implementations to get this right.
            # As a sanity check, we require `hparams` to be `None` in this case.
            if hparams is not None:
                raise ValueError(
                    "`self._hparams` already exists. Argument `hparams` "
                    "must be set to `None` in this case.")

    @abstractmethod
    def call(self, inputs, *args, **kwargs):
        r"""Defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def default_hparams():
        r"""Returns a `dict` of hyperparameters of the module with default
        values. Used to replace the missing values of input `hparams`
        during module construction.

        .. code-block:: python

            {
                "name": "module"
            }
        """
        return {
            "name": "module"
        }

    @property
    def hparams(self):
        r"""An :class:`~texar.tf.HParams` instance. The hyperparameters
        of the module.
        """
        return self._hparams
