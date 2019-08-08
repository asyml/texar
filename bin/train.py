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
"""Main script for model training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile
import yaml

import tensorflow as tf

from texar.tf import utils
from texar.tf.run import Executor


tf.flags.DEFINE_string("config_paths", "",
                       "Paths to configuration files. This can be a path to a "
                       "directory in which all files are loaded, or paths to "
                       "multiple files separated by commas. Setting a key in "
                       "these files is equivalent to setting the FLAG value "
                       "with the same name. If a key is set in both config "
                       "files and FLAG, the value in config files is used.")

tf.flags.DEFINE_string("model", "",
                       "Name of the model class.")
tf.flags.DEFINE_string("model_hparams", "{}",
                       "YAML configuration string for the model "
                       "hyper-parameters.")

tf.flags.DEFINE_string("data_hparams_train", "{}",
                       "YAML configuration string for the training data "
                       "hyper-parameters.")
tf.flags.DEFINE_string("data_hparams_eval", "{}",
                       "YAML configuration string for the evaluation data "
                       "hyper-parameters.")

tf.flags.DEFINE_integer("max_train_steps", None,
                        "Maximum number of training steps to run. "
                        "If None, train forever or until the train data "
                        "generates the OutOfRange exception. If OutOfRange "
			"occurs in the middle, training stops before "
			"max_train_steps steps.")
tf.flags.DEFINE_integer("eval_steps", None,
                        "Maximum number of evaluation steps to run. "
                        "If None, evaluate until the eval data raises an "
                        "OutOfRange exception.")

# RunConfig
tf.flags.DEFINE_string("model_dir", None,
                       "The directory where model parameters, graph, "
                       "summeries, etc are saved. If None, a local temporary "
                       "directory is created.")
tf.flags.DEFINE_integer("tf_random_seed", None,
                        "Random seed for TensorFlow initializers. Setting "
                        "this value allows consistency between reruns.")
tf.flags.DEFINE_integer("save_summary_steps", 100,
                        "Save summaries every this many steps.")
tf.flags.DEFINE_integer("save_checkpoints_steps", None,
                        "Save checkpoints every this many steps. "
                        "Can not be specified with save_checkpoints_secs.")
tf.flags.DEFINE_integer("save_checkpoints_secs", None,
                        "Save checkpoints every this many seconds. "
                        "Can not be specified with save_checkpoints_steps. "
                        "Defaults to 600 seconds if both "
                        "save_checkpoints_steps and save_checkpoints_secs "
                        "are not set. If both are set to -1, then "
                        "checkpoints are disabled.")
tf.flags.DEFINE_integer("keep_checkpoint_max", 5,
                        "Maximum number of recent checkpoint files to keep. "
                        "As new files are created, older files are deleted. "
                        "If None or 0, all checkpoint files are kept.")
tf.flags.DEFINE_integer("keep_checkpoint_every_n_hours", 10000,
                        "Number of hours between each checkpoint to be saved. "
                        "The default value of 10,000 hours effectively "
                        "disables the feature.")
tf.flags.DEFINE_integer("log_step_count_steps", 100,
                        "The frequency, in number of global steps, that the "
                        "global step/sec and the loss will be logged during "
                        "training.")
# Session config
tf.flags.DEFINE_float("per_process_gpu_memory_fraction", 1.0,
                      "Fraction of the available GPU memory to allocate for "
                      "each process.")
tf.flags.DEFINE_boolean("gpu_allow_growth", False,
                        "If true, the allocator does not pre-allocate the "
                        "entire specified GPU memory region, instead starting "
                        "small and growing as needed.")
tf.flags.DEFINE_boolean("log_device_placement", False,
                        "Whether device placements should be logged.")

FLAGS = tf.flags.FLAGS

def _process_config():
    # Loads configs
    config = utils.load_config(FLAGS.config_paths)

    # Parses YAML FLAGS
    FLAGS.model_hparams = yaml.load(FLAGS.model_hparams)
    FLAGS.data_hparams_train = yaml.load(FLAGS.data_hparams_train)
    FLAGS.data_hparams_eval = yaml.load(FLAGS.data_hparams_eval)

    # Merges
    final_config = {}
    for flag_key in dir(FLAGS):
        if flag_key in {'h', 'help', 'helpshort'}: # Filters out help flags
            continue
        flag_value = getattr(FLAGS, flag_key)
        config_value = config.get(flag_key, None)
        if isinstance(flag_value, dict) and isinstance(config_value, dict):
            final_config[flag_key] = utils.dict_patch(config_value, flag_value)
        elif flag_key in config:
            final_config[flag_key] = config_value
        else:
            final_config[flag_key] = flag_value

    # Processes
    if final_config['model_dir'] is None:
        final_config['model_dir'] = tempfile.mkdtemp()

    if final_config['save_checkpoints_steps'] is None \
            and final_config['save_checkpoints_secs'] is None:
        final_config['save_checkpoints_secs'] = 600
    if final_config['save_checkpoints_steps'] == -1 \
            and final_config['save_checkpoints_secs'] == -1:
        final_config['save_checkpoints_steps'] = None
        final_config['save_checkpoints_secs'] = None

    tf.logging.info("Final Config:\n%s", yaml.dump(final_config))

    return final_config

def _get_run_config(config):
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=\
                config['per_process_gpu_memory_fraction'],
        allow_growth=config['gpu_allow_growth'])
    sess_config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=config['log_device_placement'])

    run_config = tf.estimator.RunConfig(
        model_dir=config['model_dir'],
        tf_random_seed=config['tf_random_seed'],
        save_summary_steps=config['save_summary_steps'],
        save_checkpoints_steps=config['save_checkpoints_steps'],
        save_checkpoints_secs=config['save_checkpoints_secs'],
        keep_checkpoint_max=config['keep_checkpoint_max'],
        keep_checkpoint_every_n_hours=config['keep_checkpoint_every_n_hours'],
        log_step_count_steps=config['log_step_count_steps'],
        session_config=sess_config)

    return run_config

def main(_):
    """The entrypoint."""

    config = _process_config()

    run_config = _get_run_config(config)

    kwargs = {
        'data_hparams': config['data_hparams_train'],
        'hparams': config['model_hparams']
    }
    model = utils.check_or_get_instance_with_redundant_kwargs(
        config['model'], kwargs=kwargs,
        module_paths=['texar.tf.models', 'texar.tf.custom'])

    data_hparams = {
        'train': config['data_hparams_train'],
        'eval': config['data_hparams_eval']
    }

    exor = Executor(
        model=model,
        data_hparams=data_hparams,
        config=run_config)

    exor.train_and_evaluate(
        max_train_steps=config['max_train_steps'],
        eval_steps=config['eval_steps'])

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
