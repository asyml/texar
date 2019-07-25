# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utility functions related to XLNet encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sys
import json
import os
import re

import tensorflow as tf

from texar.data.data_utils import maybe_download

__all__ = [
    'init_from_checkpoint',
    'load_pretrained_xlnet',
    'transform_xlnet_to_texar_config'
]

_XLNET_PATH = "https://storage.googleapis.com/xlnet/released_models/"
_MODEL2URL = {
    'xlnet-large-cased': _XLNET_PATH + "cased_L-24_H-1024_A-16.zip",
    'xlnet-base-cased': _XLNET_PATH + "cased_L-12_H-768_A-12.zip"
}


def init_from_checkpoint(init_checkpoint_dir, global_vars=False):
    tvars = tf.global_variables() if global_vars else tf.trainable_variables()
    tf.logging.info("Initialize from the ckpt {}".format(init_checkpoint_dir))
    init_checkpoint = os.path.join(init_checkpoint_dir, 'xlnet_model.ckpt')
    (assignment_map, initialized_variable_names
     ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # Log customized initialization
    tf.logging.info("**** Global Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name_to_variable[name]
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return assignment_map, initialized_variable_names


def _default_download_dir():
    """
    Return the directory to which packages will be downloaded by default.
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

    return os.path.join(texar_download_dir, "xlnet")


def load_pretrained_xlnet(pretrained_model_name, cache_dir=None):
    """
    Return the directory in which the pretrained model is cached.
    """
    if pretrained_model_name in _MODEL2URL:
        download_path = _MODEL2URL[pretrained_model_name]
    else:
        raise ValueError(
            "Pre-trained model not found: {}".format(pretrained_model_name))

    if cache_dir is None:
        cache_dir = _default_download_dir()

    file_name = download_path.split('/')[-1]
    # this is required because of the way xlnet model is bundled
    file_name = "xlnet_" + file_name

    cache_path = os.path.join(cache_dir, file_name.split('.')[0])
    if not os.path.exists(cache_path):
        maybe_download(download_path, cache_dir, extract=True)
    else:
        print("Using cached pre-trained model {} from: {}".format(
            pretrained_model_name, cache_dir))

    return cache_path


def transform_xlnet_to_texar_config(config_dir):
    """
    Load the Json config file and transform it into Texar style configuration.
    """
    config_ckpt = json.loads(
        open(os.path.join(config_dir, 'xlnet_config.json')).read())
    config = dict(untie_r=config_ckpt["untie_r"],
                  num_layers=config_ckpt["n_layer"],
                  # layer
                  head_dim=config_ckpt["d_head"],
                  hidden_dim=config_ckpt["d_model"],
                  num_heads=config_ckpt["n_head"],
                  vocab_size=config_ckpt["n_token"],
                  activation="gelu",
                  ffn_inner_dim=config_ckpt["d_inner"])

    return config
