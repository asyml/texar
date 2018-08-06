"""Checkpoint averaging script."""

# This script is modified version of
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/bin/t2t_avg_all.py
# which comes with the following license and copyright notice:

# Copyright 2017 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import six

import tensorflow as tf
import numpy as np


def main():
  tf.logging.set_verbosity(tf.logging.INFO)

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--model_dir", required=True,
                      help="The model directory containing the checkpoints.")
  parser.add_argument("--output_dir", required=True,
                      help="The output directory where the averaged checkpoint will be saved.")
  parser.add_argument("--max_count", type=int, default=8,
                      help="The maximal number of checkpoints to average.")
  args = parser.parse_args()

  if args.model_dir == args.output_dir:
    raise ValueError("Model and output directory must be different")

  checkpoints_path = tf.train.get_checkpoint_state(args.model_dir).all_model_checkpoint_paths
  if len(checkpoints_path) > args.max_count:
    checkpoints_path = checkpoints_path[-args.max_count:]
  num_checkpoints = len(checkpoints_path)

  tf.logging.info("Averaging %d checkpoints..." % num_checkpoints)
  tf.logging.info("Listing variables...")

  var_list = tf.train.list_variables(checkpoints_path[0])
  avg_values = {}
  for name, shape in var_list:
    if not name.startswith("global_step"):
      avg_values[name] = np.zeros(shape)

  for checkpoint_path in checkpoints_path:
    tf.logging.info("Loading checkpoint %s" % checkpoint_path)
    reader = tf.train.load_checkpoint(checkpoint_path)
    for name in avg_values:
      avg_values[name] += reader.get_tensor(name) / num_checkpoints

  tf_vars = []
  for name, value in six.iteritems(avg_values):
    tf_vars.append(tf.get_variable(name, shape=value.shape))
  placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
  assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]

  latest_step = int(checkpoints_path[-1].split("-")[-1])
  out_base_file = os.path.join(args.output_dir, "model.ckpt")
  global_step = tf.get_variable(
      "global_step",
      initializer=tf.constant(latest_step, dtype=tf.int64),
      trainable=False)
  saver = tf.train.Saver(tf.global_variables())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for p, assign_op, (name, value) in zip(placeholders, assign_ops, six.iteritems(avg_values)):
      sess.run(assign_op, {p: value})
    tf.logging.info("Saving averaged checkpoint to %s-%d" % (out_base_file, latest_step))
    saver.save(sess, out_base_file, global_step=global_step)


if __name__ == "__main__":
  main()
