#
"""
Discriminators.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.modules.module_base import ModuleBase

__all__ = [
]

# TODO(zhiting): Discriminators should be based on Encoders, e.g., takes in
# an encoder as `feature_extractor`, and add an extra classification layer
# on the encoder outputs.
class Discriminator(ModuleBase):
    raise NotImplementedError
