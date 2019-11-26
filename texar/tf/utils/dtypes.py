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
Utility functions related to data types.
"""

__all__ = [
    "is_callable",
    "is_str",
]


def is_callable(x):
    r"""Return `True` if :attr:`x` is callable.
    """
    return callable(x)


def is_str(x):
    r"""Returns `True` if :attr:`x` is either a str or unicode.
    Returns `False` otherwise.
    """
    return isinstance(x, str)
