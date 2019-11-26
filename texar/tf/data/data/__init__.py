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
Modules of texar library data inputs.
"""

# pylint: disable=wildcard-import

from texar.tf.data.data.data_base import *
from texar.tf.data.data.scalar_data import *
from texar.tf.data.data.text_data_base import *
from texar.tf.data.data.mono_text_data import *
from texar.tf.data.data.paired_text_data import *
from texar.tf.data.data.multi_aligned_data import *
from texar.tf.data.data.data_iterators import *
from texar.tf.data.data.dataset_utils import *
from texar.tf.data.data.tfrecord_data import *
