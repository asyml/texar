# -*- coding: utf-8 -*-
#
"""
Unit tests for data related operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import copy
import shutil
import tempfile
import tensorflow as tf
import texar as tx


class TFRecordDataTest(tf.test.TestCase):
    """Tests tfrecord data class.
    """
    # pylint: disable=too-many-locals
    def setUp(self):
        tf.test.TestCase.setUp(self)

        # Create test data
        # pylint: disable=no-member
        self._test_dir = tempfile.mkdtemp()

        cat_in_snow = tf.keras.utils.get_file(
            os.path.join(self._test_dir, 'cat_0.jpg'),
            'https://storage.googleapis.com/download.tensorflow.org/'
            'example_images/320px-Felis_catus-cat_on_snow.jpg')
        williamsburg_bridge = tf.keras.utils.get_file(
            os.path.join(self._test_dir, 'bridge_0.jpg'),
            'https://storage.googleapis.com/download.tensorflow.org/'
            'example_images/194px-New_East_River_Bridge_from_Brooklyn_'
            'det.4a09796u.jpg')

        def _bytes_feature(value=None):
            """Returns a bytes_list from a string / byte.
            """
            # pylint: disable=undefined-loop-variable
            value = tf.compat.as_bytes(
                value,
                encoding='utf-8'
            )
            return tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[value]))

        def _int64_feature(value=None):
            """Returns an int64_list from a bool / enum / int / uint.
            """
            return tf.train.Feature(
                int64_list=tf.train.Int64List(value=[value]))

        _feature_original_types = {
            'height': ['tf.int64', 'FixedLenFeature'],
            'width': ['tf.int64', 'FixedLenFeature'],
            'label': ['tf.int64', 'FixedLenFeature'],
            'shape': [tf.int64, 'VarLenFeature'],
            'image_raw': ['tf.string', 'FixedLenFeature'],
            'variable1': [tf.string, 'FixedLenFeature'],
            'variable2': ['tf.int64', 'FixedLenFeature'],
        }
        self._feature_convert_types = {
            'variable1': 'tf.float32',
            'variable2': 'tf.string',
        }
        _image_options = {}
        self._unconvert_features = ['height', 'width', 'label']
        def _image_example(image_string, image_shape, label):
            """Create data example with image
            """
            feature = {
                'height': _int64_feature(image_shape[0]),
                'width': _int64_feature(image_shape[1]),
                'shape': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=list(image_shape))),
                'label': _int64_feature(label),
                'image_raw': _bytes_feature(image_string),
                'variable1': _bytes_feature('1234567890'),
                'variable2': _int64_feature(9876543210),
            }
            return tf.train.Example(
                features=tf.train.Features(feature=feature))

        self._dataset_valid = {
            'height': [],
            'width': [],
            'shape': [],
            'label': [],
            'image_raw': [],
            'variable1': [],
            'variable2': [],
        }
        _toy_image_labels_valid = {
            cat_in_snow : 0,
            williamsburg_bridge : 1,
        }
        _toy_image_shapes = {
            cat_in_snow: (213, 320, 3),
            williamsburg_bridge: (239, 194),
        }
        _tfrecord_filepath = os.path.join(
            self._test_dir,
            'test.tfrecord')
        # Prepare Validation data
        with tf.python_io.TFRecordWriter(_tfrecord_filepath) as writer:
            for image_path, label in _toy_image_labels_valid.items():

                with open(image_path, 'rb') as fid:
                    image_data = fid.read()
                image_shape = _toy_image_shapes[image_path]

                tf_example = _image_example(image_data, image_shape, label)
                writer.write(tf_example.SerializeToString())

                #_construct_dataset_valid("", shape, label)
                single_data = {
                    'height': image_shape[0],
                    'width': image_shape[1],
                    'shape': image_shape,
                    'label': label,
                    'image_raw': image_data,
                    'variable1': "1234567890",
                    'variable2': int(9876543210),
                }
                for key, value in single_data.items():
                    self._dataset_valid[key].append(value)

        self._hparams = {
            "num_epochs": 1,
            "batch_size": 1,
            "shuffle": False,
            "dataset": {
                "files": _tfrecord_filepath,
                "feature_original_types": _feature_original_types,
                "feature_convert_types": self._feature_convert_types,
                "image_options": [_image_options],
            }
        }

    def tearDown(self):
        """Remove the downloaded files after the test
        """
        shutil.rmtree(self._test_dir)

    def _run_and_test(self, hparams):
        # Construct database
        tfrecord_data = tx.data.TFRecordData(hparams)
        iterator = tfrecord_data.dataset.make_initializable_iterator()
        data_batch = iterator.get_next()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            sess.run(iterator.initializer)
            i = 0
            def _prod(lst):
                res = 1
                for i in lst:
                    res *= i
                return res
            while True:
                try:
                    # Run the logics
                    data_batch_ = sess.run(data_batch)
                    self.assertEqual(
                        set(data_batch_.keys()),
                        set(tfrecord_data.list_items()))

                    # Check data consistency
                    for key in self._unconvert_features:
                        value = data_batch_[key][0]
                        self.assertEqual(value, self._dataset_valid[key][i])
                    self.assertEqual(
                        list(data_batch_['shape'].values),
                        list(self._dataset_valid['shape'][i]))

                    # Check data type conversion
                    for key, item in self._feature_convert_types.items():
                        value = data_batch_[key][0]
                        if item == 'tf.string' or item is tf.string:
                            self.assertTrue(isinstance(value, bytes))
                        else:
                            dtype_matched = (
                                tx.utils.dtypes.get_tf_dtype(str(value.dtype))
                                is tx.utils.dtypes.get_tf_dtype(item))
                            self.assertTrue(dtype_matched)

                    # Check image decoding and resize
                    if hparams["dataset"].get("image_options"):
                        image_options = hparams["dataset"].get("image_options")
                        if isinstance(image_options, dict):
                            image_options = [image_options]
                        for image_option_feature in image_options:
                            image_key = image_option_feature.get(
                                "image_feature_name")
                            if image_key is None:
                                continue
                            image_gen = data_batch_[image_key][0]
                            image_valid_shape = self._dataset_valid["shape"][i]
                            resize_height = image_option_feature.get(
                                "resize_height")
                            resize_width = image_option_feature.get(
                                "resize_width")
                            if resize_height and resize_width:
                                self.assertEqual(
                                    image_gen.shape[0] * image_gen.shape[1],
                                    resize_height * resize_width)
                            else:
                                self.assertEqual(
                                    _prod(image_gen.shape),
                                    _prod(image_valid_shape))
                    i += 1
                except tf.errors.OutOfRangeError:
                    print('Done -- epoch limit reached')
                    break

    def test_default_setting(self):
        """Tests the logics of TFRecordData.
        """
        self._run_and_test(self._hparams)

    def test_image_resize(self):
        """Tests the image resize function
        """
        hparams = copy.copy(self._hparams)
        _image_options = {
            'image_feature_name': 'image_raw',
            'resize_height': 512,
            'resize_width': 512,
        }
        hparams["dataset"].update({"image_options": _image_options})
        self._run_and_test(hparams)

if __name__ == "__main__":
    tf.test.main()
