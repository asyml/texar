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
"""Downloads data.
"""
import texar as tx

# pylint: disable=invalid-name

def prepare_data():
    """Downloads data.
    """
    tx.data.maybe_download(
        urls='https://drive.google.com/file/d/'
             '1HaUKEYDBEk6GlJGmXwqYteB-4rS9q8Lg/view?usp=sharing',
        path='./',
        filenames='yelp.zip',
        extract=True)

def main():
    """Entrypoint.
    """
    prepare_data()

if __name__ == '__main__':
    main()
