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
Utils of Pre-trained Modules.
"""

import os
import sys


__all__ = [
    "default_download_dir",
]


def default_download_dir(name):
    r"""Return the directory to which packages will be downloaded by default.
    """
    package_dir = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))))
    if os.access(package_dir, os.W_OK):
        texar_download_dir = os.path.join(package_dir, 'texar_download')
    else:
        # On Windows, use %APPDATA%
        if sys.platform == 'win32' and 'APPDATA' in os.environ:
            home_dir = os.environ['APPDATA']

        # Otherwise, install in the user's home directory.
        else:
            home_dir = os.path.expanduser('~/')
            if home_dir == '~/':
                raise ValueError("Could not find a default download directory")

        texar_download_dir = os.path.join(home_dir, 'texar_download')

    if not os.path.exists(texar_download_dir):
        os.mkdir(texar_download_dir)

    return os.path.join(texar_download_dir, name)
