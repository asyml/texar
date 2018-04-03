#
"""
Various text tokenizers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

# pylint: disable=too-many-instance-attributes, too-many-arguments,
# pylint: disable=no-member, invalid-name

__all__ = [
]

class TokenizerBase(object):
    """Base class for all tokenizers.
    """

    def tokenize(self, text):
        """Tokenizes text.

        Args:
            text: A Tensor or string to tokenize.

        Returns:
            A 1D string Tensor if :attr:`text` is Tensor, a list of unicode
            strings if :attr:`text` is string.
        """
        if tf.contrib.framework.is_tensor(text):
            return self._tokenize_tensor(text)
        else:
            return self._tokenize_string(tf.compat.as_text(text))
