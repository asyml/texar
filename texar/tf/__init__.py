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
Modules of texar library.
"""

# pylint: disable=wildcard-import

import pkg_resources
import tensorflow as tf

VERSION_WARNING = "1.13.2"


if (pkg_resources.parse_version(tf.__version__) <=
        pkg_resources.parse_version(VERSION_WARNING)):
    tf.logging.set_verbosity(tf.logging.ERROR)
else:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from texar.tf.version import VERSION as __version__

from texar.tf import agents
from texar.tf import core
from texar.tf import data
from texar.tf import evals
from texar.tf import losses
from texar.tf import models
from texar.tf import modules
from texar.tf import run
from texar.tf import utils
from texar.tf.module_base import *
from texar.tf.hyperparams import *
from texar.tf.context import *
