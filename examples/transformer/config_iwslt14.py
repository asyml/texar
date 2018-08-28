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
wbatchsize = 2048
filename_prefix = "processed."
test_batch_size = 64
eval_steps = 2000
max_train_epoch = 20
max_training_steps = 100000
max_decode_len = 256
input_dir = 'temp/run_en_vi_spm/data'
vocab_file = input_dir + '/processed.vocab.pickle'
