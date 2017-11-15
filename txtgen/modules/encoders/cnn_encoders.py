#
"""
Various CNN encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from txtgen.modules.encoders.encoder_base import EncoderBase
from txtgen.core import layers

# pylint: disable=not-context-manager, too-many-arguments

__all__ = [
]

class CNNEncoderBase(EncoderBase):
    """Base class for all CNN encoder classes."""
    raise NotImplementedError
